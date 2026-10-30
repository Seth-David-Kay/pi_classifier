/*
 * Bird sound classifier: a small fully-connected neural network written from scratch in C.
 *
 * Species (10 common birds around Bethesda, MD): American Robin, Northern Cardinal,
 * Carolina Chickadee, Tufted Titmouse, Blue Jay, Carolina Wren, Mourning Dove,
 * American Crow, Northern Mockingbird, House Sparrow.
 *
 * Architecture: 512 inputs -> 128 (ReLU) -> 64 (ReLU) -> 10 (softmax), trained with
 * cross-entropy loss, minibatch SGD with momentum, and He-initialized weights.
 * The 512 inputs are a 32 mel-band x 16 time-bin log-mel spectrogram of a 3 second segment.
 *
 * Data: `download` asks the Xeno-canto API v3 (via the curl command, needs XC_API_KEY set)
 * for quality-A recordings of each species, parses the JSON reply with cJSON, and downloads
 * each MP3 into data/mp3/. A manifest (data/manifest.txt) lists "<species epithet> <path>" per clip.
 * Each MP3 is decoded in-process with minimp3, mixed to mono, and resampled to 16 kHz.
 * It is cut into up to four 3 second segments, each run through a Hann-windowed 512-point
 * FFT, a triangular mel filterbank, log(1 + x), and pooled in time to the 32x16 grid.
 * Clips (not segments) are split 60/20/20 into train/validation/test so no clip leaks across sets.
 *
 * Memory: everything allocates from one base-library Arena; temporary work is rolled back
 * with arena marks. Text handling, parsing, file IO, hash maps, arrays, and randomness all
 * come from the base library.
 *
 * Build: gcc -Wall -Wextra -O2 main.c -o birds -lm
 * Use:   XC_API_KEY=... ./birds download | ./birds train | ./birds infer <file.mp3|file.wav>
 */

#define MINIMP3_IMPLEMENTATION
#include "deps/minimp3_ex.h"
#include "deps/cJSON.c"
#include "base/base_all.h"

#define SR 16000
#define NFFT 512
#define HOP 256
#define NMEL 32
#define TBINS 16
#define SEG_LEN 48000
#define MAX_SEGS 4
#define MAX_SECONDS 12
#define IN (NMEL * TBINS)
#define H1 128
#define H2 64
#define OUT 4 // number of birbs
#define NPARAMS (H1 * IN + H1 + H2 * H1 + H2 + OUT * H2 + OUT)
#define CLIPS_PER_SPECIES 5
#define SEED 42
#define MODEL_PATH "bird_model.bin"
#define MANIFEST_PATH "data/manifest.txt"

structdef(Bird) { String name, genus, epithet; };
structdef_end(Bird)

typedef struct {
    f32 W1[H1 * IN], b1[H1];
    f32 W2[H2 * H1], b2[H2];
    f32 W3[OUT * H2], b3[OUT];
    f32 mean[IN], std[IN];
} Model;

typedef struct {
    f32 z1[H1], a1[H1], z2[H2], a2[H2], z3[OUT], p[OUT];
} Cache;

typedef struct {
    f32 *x;
    i32 *y;
    i32 *clip;
    i32 n, cap;
} Dataset;

typedef struct {
    Dataset train, val, test;
} Split;

typedef struct {
    f32 acc, loss;
} Eval;

typedef struct {
    f32 *s;
    i32 n;
} Audio;

typedef struct {
    f32 *hann;
    f32 *fb;
} Mel;

typedef struct {
    Model *model;
    Array_String names;
} Loaded;

// Builds the list of the ten target species with their scientific names.
static Array_Bird make_birds(Arena *arena) {
    const char *rows[OUT][3] = {
        {"American Robin", "Turdus", "migratorius"},
        {"Northern Cardinal", "Cardinalis", "cardinalis"},
        {"Carolina Chickadee", "Poecile", "carolinensis"},
        // {"Tufted Titmouse", "Baeolophus", "bicolor"},
        // {"Blue Jay", "Cyanocitta", "cristata"},
        // {"Carolina Wren", "Thryothorus", "ludovicianus"},
        // {"Mourning Dove", "Zenaida", "macroura"},
        // {"American Crow", "Corvus", "brachyrhynchos"},
        // {"Northern Mockingbird", "Mimus", "polyglottos"},
        {"House Sparrow", "Passer", "domesticus"}
    };
    Array_Bird birds = {0};
    birds.arena = arena;
    for (i32 i = 0; i < OUT; i++) {
        Bird b = {string_from_cstr(rows[i][0]), string_from_cstr(rows[i][1]), string_from_cstr(rows[i][2])};
        array_push(&birds, b);
    }
    return birds;
}

// Copies a String into a NUL-terminated C string allocated from the arena.
static char *cstr(Arena *arena, String s) {
    char *out = arena_make_type(arena, s.count + 1, char);
    memcpy(out, s.data, s.count);
    out[s.count] = 0;
    return out;
}

// Runs a shell command and returns true if it exited with status 0.
static bool run(Arena *arena, String cmd) {
    return system(cstr(arena, cmd)) == 0;
}

// Returns the size of a file in bytes, or -1 if it does not exist.
static i64 file_size(char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    i64 len = ftell(f);
    fclose(f);
    return len;
}

// Downloads one URL to a file with curl, returning true if a plausible file was written.
static bool fetch_clip(Arena *arena, char *url, char *out_path) {
    ArenaMark mark = arena_mark(arena);
    String cmd = string_format(arena, "curl -s -L -f -o \"%s\" \"%s\"", out_path, url);
    bool ok = run(arena, cmd) && file_size(out_path) >= 2000;
    arena_pop_to(arena, mark);
    if (!ok) remove(out_path);
    return ok;
}

// Queries Xeno-canto for one species, downloads up to CLIPS_PER_SPECIES MP3s, and returns manifest lines.
static Array_String download_species(Arena *arena, Bird bird, i32 idx, char *key) {
    Array_String lines = {0};
    lines.arena = arena;
    i32 per_page = CLIPS_PER_SPECIES < 50 ? 50 : CLIPS_PER_SPECIES;
    char *meta = cstr(arena, string_format(arena, "data/meta_%d.json", (int)idx));
    String cmd = string_format(arena,
        "curl -s -L -o \"%s\" \"https://xeno-canto.org/api/3/recordings?query=gen:%s+sp:%s+q:A&per_page=%d&key=%s\"",
        meta, cstr(arena, bird.genus), cstr(arena, bird.epithet), (int)per_page, key);
    if (!run(arena, cmd)) return lines;

    String json;
    if (!file_read_all(arena, meta, &json)) return lines;
    cJSON *root = cJSON_ParseWithLength((const char *)json.data, json.count);
    if (!root) {
        println_string(string_format(arena, "  could not parse API reply for %s", cstr(arena, bird.name)));
        return lines;
    }
    cJSON *err = cJSON_GetObjectItemCaseSensitive(root, "error");
    if (cJSON_IsString(err)) println_string(string_format(arena, "  API error: %s", err->valuestring));

    cJSON *rec;
    cJSON *recs = cJSON_GetObjectItemCaseSensitive(root, "recordings");
    cJSON_ArrayForEach(rec, recs) {
        if (lines.count >= CLIPS_PER_SPECIES) break;
        cJSON *id = cJSON_GetObjectItemCaseSensitive(rec, "id");
        cJSON *file = cJSON_GetObjectItemCaseSensitive(rec, "file");
        if (!cJSON_IsString(id) || !cJSON_IsString(file)) continue;
        u64 id_num = 0;
        if (!parse_u64(string_from_cstr(id->valuestring), &id_num)) continue;
        if (string_find(string_from_cstr(file->valuestring), string_from_cstr("https://xeno-canto.org/")) != 0) continue;
        if (strpbrk(file->valuestring, "\"`$\\ ")) continue;
        char *path = cstr(arena, string_format(arena, "data/mp3/%d_%llu.mp3", (int)idx, (unsigned long long)id_num));
        if (file_size(path) < 2000 && !fetch_clip(arena, file->valuestring, path)) continue;
        String line = string_format(arena, "%s %s", cstr(arena, bird.epithet), path);
        array_push(&lines, line);
    }
    cJSON_Delete(root);
    return lines;
}

// Downloads clips for all ten species, skipping files already on disk, and writes the manifest.
static bool download_all_data(Arena *arena, Array_Bird birds) {
    if (!run(arena, string_from_cstr("command -v curl > /dev/null 2>&1"))) {
        println_string(string_from_cstr("curl was not found on PATH; install it to download data."));
        return false;
    }
    char *key = getenv("XC_API_KEY");
    if (!key) {
        println_string(string_from_cstr("Set XC_API_KEY to your Xeno-canto API key (https://xeno-canto.org/account)."));
        return false;
    }
    if (!run(arena, string_from_cstr("mkdir -p data/mp3"))) return false;

    StringBuilder sb;
    sb_init(&sb, 4096);
    for (u64 i = 0; i < birds.count; i++) {
        println_string(string_format(arena, "Downloading %s...", cstr(arena, birds.data[i].name)));
        Array_String lines = download_species(arena, birds.data[i], (i32)i, key);
        println_string(string_format(arena, "  %d clips ready", (int)lines.count));
        for (u64 j = 0; j < lines.count; j++) {
            sb_push_string(&sb, lines.data[j]);
            sb_push(&sb, '\n');
        }
    }
    bool ok = file_write_all(MANIFEST_PATH, sb_build(&sb));
    sb_free(&sb);
    return ok;
}

// Reads a little-endian 32-bit unsigned integer from bytes.
static u32 rd_u32(u8 *p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

// Reads a little-endian 16-bit unsigned integer from bytes.
static u16 rd_u16(u8 *p) {
    return (u16)(p[0] | (p[1] << 8));
}

// Resamples to 16 kHz by averaging each output sample's source window: y[i] = mean(x[i*r .. (i+1)*r]).
static Audio resample(Arena *arena, Audio in, i32 hz) {
    ASSERT_MSG(hz > 0, "invalid sample rate %d", (int)hz);
    if (hz == SR || in.n <= 0) return in;
    f64 ratio = (f64)hz / SR;
    i32 n = (i32)(in.n / ratio);
    f32 *out = arena_make_type(arena, n + 1, f32);
    for (i32 i = 0; i < n; i++) {
        i32 a = (i32)(i * ratio), b = (i32)((i + 1) * ratio);
        if (b >= in.n) b = in.n - 1;
        if (b < a) b = a;
        f32 sum = 0;
        for (i32 j = a; j <= b; j++) sum += in.s[j];
        out[i] = sum / (f32)(b - a + 1);
    }
    return (Audio){out, n};
}

// Decodes an MP3 with minimp3 into mono floats in [-1, 1] at 16 kHz.
static Audio load_mp3(Arena *arena, char *path) {
    mp3dec_t dec;
    mp3dec_file_info_t info;
    mp3dec_init(&dec);
    if (mp3dec_load(&dec, path, &info, NULL, NULL) != 0 || !info.buffer || info.channels < 1) return (Audio){0};
    i32 ch = info.channels, hz = info.hz;
    i64 frames = (i64)(info.samples / (u64)ch);
    i64 cap = (i64)MAX_SECONDS * hz;
    if (frames > cap) frames = cap;
    if (frames <= 0) {
        free(info.buffer);
        return (Audio){0};
    }
    f32 *mono = arena_make_type(arena, frames, f32);
    for (i64 i = 0; i < frames; i++) {
        f32 s = 0;
        for (i32 c = 0; c < ch; c++) s += info.buffer[i * ch + c];
        mono[i] = s / (32768.0f * ch);
    }
    free(info.buffer);
    return resample(arena, (Audio){mono, (i32)frames}, hz);
}

// Parses a 16-bit PCM WAV file into mono floats in [-1, 1] at 16 kHz.
static Audio load_wav(Arena *arena, char *path) {
    String file;
    if (!file_read_all(arena, path, &file)) return (Audio){0};
    u8 *buf = file.data;
    i64 len = (i64)file.count;
    if (len < 44 || memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0) return (Audio){0};
    i32 channels = 0, hz = 0, bits = 0, fmt_tag = 0;
    u8 *data = NULL;
    i64 data_len = 0, pos = 12;
    while (pos + 8 <= len) {
        u32 sz = rd_u32(buf + pos + 4);
        u8 *body = buf + pos + 8;
        if (memcmp(buf + pos, "fmt ", 4) == 0 && pos + 8 + 16 <= len) {
            fmt_tag = rd_u16(body);
            channels = rd_u16(body + 2);
            hz = (i32)rd_u32(body + 4);
            bits = rd_u16(body + 14);
        } else if (memcmp(buf + pos, "data", 4) == 0) {
            data = body;
            data_len = (i64)sz;
            if (pos + 8 + data_len > len) data_len = len - pos - 8;
            break;
        }
        pos += 8 + (i64)sz + (sz & 1);
    }
    if (!data || fmt_tag != 1 || bits != 16 || channels < 1 || hz <= 0) return (Audio){0};
    i64 frames = data_len / (2 * channels);
    i64 cap = (i64)MAX_SECONDS * hz;
    if (frames > cap) frames = cap;
    if (frames <= 0) return (Audio){0};
    f32 *mono = arena_make_type(arena, frames, f32);
    for (i64 i = 0; i < frames; i++) {
        f32 s = 0;
        for (i32 c = 0; c < channels; c++) s += (i16)rd_u16(data + (i * channels + c) * 2);
        mono[i] = s / (32768.0f * channels);
    }
    return resample(arena, (Audio){mono, (i32)frames}, hz);
}

// Loads an MP3 or WAV file, chosen by its header bytes rather than its extension, into mono 16 kHz floats.
static Audio load_audio(Arena *arena, char *path) {
    u8 h[4] = {0};
    FILE *f = fopen(path, "rb");
    if (!f) return (Audio){0};
    u64 got = fread(h, 1, 4, f);
    fclose(f);
    if (got < 4) return (Audio){0};
    if (memcmp(h, "RIFF", 4) == 0) return load_wav(arena, path);
    if (memcmp(h, "ID3", 3) == 0 || (h[0] == 0xFF && (h[1] & 0xE0) == 0xE0)) return load_mp3(arena, path);
    return (Audio){0};
}

// In-place radix-2 FFT: X[k] = sum_n x[n] * e^(-2*pi*i*k*n/N).
static void fft(f32 *re, f32 *im, i32 n) {
    for (i32 i = 1, j = 0; i < n; i++) {
        i32 bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            f32 t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }
    for (i32 len = 2; len <= n; len <<= 1) {
        f32 ang = -6.2831853f / (f32)len;
        f32 wr = cosf(ang), wi = sinf(ang);
        for (i32 i = 0; i < n; i += len) {
            f32 cr = 1.0f, ci = 0.0f;
            for (i32 k = 0; k < len / 2; k++) {
                i32 a = i + k, b = i + k + len / 2;
                f32 tr = re[b] * cr - im[b] * ci;
                f32 ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                f32 nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
    }
}

// Converts Hz to mel: mel = 2595 * log10(1 + f / 700).
static f32 hz_to_mel(f32 f) {
    return 2595.0f * log10f(1.0f + f / 700.0f);
}

// Converts mel to Hz: f = 700 * (10^(mel / 2595) - 1).
static f32 mel_to_hz(f32 m) {
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

// Builds the Hann window w[n] = 0.5 - 0.5 cos(2 pi n / (N - 1)) and the triangular mel filterbank.
static Mel build_mel_filterbank(Arena *arena) {
    i32 bins = NFFT / 2 + 1;
    Mel mel = {arena_make_type(arena, NFFT, f32), arena_make_type(arena, NMEL * bins, f32)};
    f32 *pts = arena_make_type(arena, NMEL + 2, f32);
    for (i32 i = 0; i < NFFT; i++) mel.hann[i] = 0.5f - 0.5f * cosf(6.2831853f * i / (NFFT - 1));
    f32 mlo = hz_to_mel(100.0f), mhi = hz_to_mel(8000.0f);
    for (i32 m = 0; m < NMEL + 2; m++) pts[m] = mel_to_hz(mlo + (mhi - mlo) * m / (NMEL + 1));
    for (i32 m = 0; m < NMEL; m++) {
        f32 l = pts[m], c = pts[m + 1], r = pts[m + 2];
        for (i32 k = 0; k < bins; k++) {
            f32 f = (f32)k * SR / NFFT, w = 0.0f;
            if (f >= l && f <= c) w = (f - l) / (c - l);
            else if (f > c && f < r) w = (r - f) / (r - c);
            mel.fb[m * bins + k] = w;
        }
    }
    return mel;
}

// Returns a 32x16 log-mel feature vector: out = log(1 + FB * |FFT(hann * x)|^2), time-pooled and mean-centered.
static f32 *extract_features(Arena *arena, Mel mel, f32 *s, i32 valid, i32 total) {
    ASSERT(total > 0);
    i32 bins = NFFT / 2 + 1;
    i32 nf = total < NFFT ? 1 : (total - NFFT) / HOP + 1;
    f32 *out = arena_make_type(arena, IN, f32);
    memset(out, 0, IN * sizeof(f32));

    ArenaMark mark = arena_mark(arena);
    f32 *re = arena_make_type(arena, NFFT, f32);
    f32 *im = arena_make_type(arena, NFFT, f32);
    f32 *pw = arena_make_type(arena, bins, f32);
    f32 *cnt = arena_make_type(arena, TBINS, f32);
    memset(cnt, 0, TBINS * sizeof(f32));
    for (i32 f = 0; f < nf; f++) {
        for (i32 i = 0; i < NFFT; i++) {
            i32 idx = f * HOP + i;
            f32 v = idx < valid ? s[idx] : 0.0f;
            re[i] = v * mel.hann[i] * 32768.0f;
            im[i] = 0.0f;
        }
        fft(re, im, NFFT);
        for (i32 k = 0; k < bins; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
        i32 t = f * TBINS / nf;
        for (i32 m = 0; m < NMEL; m++) {
            f32 e = 0.0f;
            for (i32 k = 0; k < bins; k++) e += mel.fb[m * bins + k] * pw[k];
            out[m * TBINS + t] += logf(1.0f + e);
        }
        cnt[t] += 1.0f;
    }
    f32 mean = 0.0f;
    for (i32 m = 0; m < NMEL; m++)
        for (i32 t = 0; t < TBINS; t++) {
            if (cnt[t] > 0.0f) out[m * TBINS + t] /= cnt[t];
            mean += out[m * TBINS + t];
        }
    mean /= IN;
    for (i32 i = 0; i < IN; i++) out[i] -= mean;
    arena_pop_to(arena, mark);
    return out;
}

// Number of 3 second segments taken from a clip of n samples: min(MAX_SEGS, n / SEG_LEN), at least 1 if n >= SEG_LEN / 4.
static i32 count_segments(i32 n) {
    i32 segs = n / SEG_LEN;
    if (segs == 0 && n >= SEG_LEN / 4) segs = 1;
    return segs > MAX_SEGS ? MAX_SEGS : segs;
}

// Extracts features for one zero-padded 3 second segment of a clip.
static f32 *segment_features(Arena *arena, Mel mel, Audio a, i32 seg) {
    i32 start = seg * SEG_LEN, avail = a.n - start;
    if (avail > SEG_LEN) avail = SEG_LEN;
    if (avail < 0) avail = 0;
    return extract_features(arena, mel, a.s + start, avail, SEG_LEN);
}

// Allocates an empty dataset with room for cap samples.
static Dataset ds_alloc(Arena *arena, i32 cap) {
    Dataset d = {0};
    d.cap = cap;
    d.x = arena_make_type(arena, (u64)cap * IN, f32);
    d.y = arena_make_type(arena, cap, i32);
    d.clip = arena_make_type(arena, cap, i32);
    return d;
}

// Returns the dataset with one more feature vector, label, and clip id appended.
static Dataset ds_push(Dataset d, f32 *feat, i32 y, i32 clip) {
    ASSERT_MSG(d.n < d.cap, "dataset full (capacity %d)", (int)d.cap);
    memcpy(d.x + (u64)d.n * IN, feat, IN * sizeof(f32));
    d.y[d.n] = y;
    d.clip[d.n] = clip;
    d.n++;
    return d;
}

// Loads every clip in the manifest, extracts features per segment, and labels them via a species hash map.
static Dataset build_dataset(Arena *arena, Mel mel, Array_Bird birds) {
    String text;
    if (!file_read_all(arena, MANIFEST_PATH, &text)) return (Dataset){0};
    Array_String lines = string_split(arena, string_trim(text), '\n');
    Dataset ds = ds_alloc(arena, (i32)lines.count * MAX_SEGS + 1);

    HashMap_String_u64 labels;
    HashMap_String_u64_init(&labels, arena);
    for (u64 i = 0; i < birds.count; i++) HashMap_String_u64_put(&labels, birds.data[i].epithet, i);

    i32 clip = 0, per_class[OUT] = {0}, per_class_clips[OUT] = {0};
    for (u64 i = 0; i < lines.count; i++) {
        ArenaMark mark = arena_mark(arena);
        Array_String parts = string_split(arena, string_trim(lines.data[i]), ' ');
        u64 label = 0;
        if (parts.count == 2 && HashMap_String_u64_get(&labels, parts.data[0], &label)) {
            Audio a = load_audio(arena, cstr(arena, parts.data[1]));
            if (a.n == 0) println_string(string_format(arena, "  skipped (could not decode): %s", cstr(arena, parts.data[1])));
            i32 segs = count_segments(a.n);
            for (i32 g = 0; g < segs; g++) {
                ds = ds_push(ds, segment_features(arena, mel, a, g), (i32)label, per_class_clips[label]);
                per_class[label]++;
            }
            if (segs > 0) {
                per_class_clips[label]++;
                clip++;
            }
        }
        arena_pop_to(arena, mark);
    }

    print("Loaded % segments from % clips\n", fmt(int, ds.n), fmt(int, clip));
    for (u64 i = 0; i < birds.count; i++) {
        ArenaMark mark = arena_mark(arena);
        println_string(string_format(arena, "  %-22s %d", cstr(arena, birds.data[i].name), (int)per_class[i]));
        arena_pop_to(arena, mark);
    }
    return ds;
}

// Splits by clip id so clip index 0 goes to validation, 1 to test, the rest to training, so about 60/20/20
static Split split_dataset(Arena *arena, Dataset all) {
    i32 nval = 0, ntest = 0;
    for (i32 i = 0; i < all.n; i++) {
        i32 b = all.clip[i] % 5;
        if (b == 0) nval++;
        else if (b == 1) ntest++;
    }
    Split sp = {ds_alloc(arena, all.n - nval - ntest + 1), ds_alloc(arena, nval + 1), ds_alloc(arena, ntest + 1)};
    for (i32 i = 0; i < all.n; i++) {
        f32 *feat = all.x + (u64)i * IN;
        i32 b = all.clip[i] % 5;
        if (b == 0) sp.val = ds_push(sp.val, feat, all.y[i], all.clip[i]);
        else if (b == 1) sp.test = ds_push(sp.test, feat, all.y[i], all.clip[i]);
        else sp.train = ds_push(sp.train, feat, all.y[i], all.clip[i]);
    }
    return sp;
}

// Z-scores one feature vector: x = (x - mean) / std.
static void apply_normalization(Model *m, f32 *x) {
    for (i32 i = 0; i < IN; i++) x[i] = (x[i] - m->mean[i]) / m->std[i];
}

// Computes per-feature mean and std on the training set and normalizes all three sets with them.
static void normalize_dataset(Model *m, Dataset train, Dataset val, Dataset test) {
    ASSERT(train.n > 0);
    for (i32 i = 0; i < IN; i++) {
        f64 mu = 0, var = 0;
        for (i32 n = 0; n < train.n; n++) mu += train.x[(u64)n * IN + i];
        mu /= train.n;
        for (i32 n = 0; n < train.n; n++) {
            f64 dlt = train.x[(u64)n * IN + i] - mu;
            var += dlt * dlt;
        }
        m->mean[i] = (f32)mu;
        m->std[i] = sqrtf((f32)(var / train.n)) + 1e-6f;
    }
    for (i32 n = 0; n < train.n; n++) apply_normalization(m, train.x + (u64)n * IN);
    for (i32 n = 0; n < val.n; n++) apply_normalization(m, val.x + (u64)n * IN);
    for (i32 n = 0; n < test.n; n++) apply_normalization(m, test.x + (u64)n * IN);
}

// Fills weights with He initialization: w ~ N(0, 2 / n_in).
static void he_fill(f32 *w, i32 count, i32 n_in, u64 seed) {
    Rng rng = rng_create(seed);
    f64 sd = sqrt(2.0 / (f64)n_in);
    for (i32 i = 0; i < count; i++) w[i] = (f32)rng_normal_f64(&rng, 0.0, sd);
}

// Allocates a zeroed model, then He-initializes all weight matrices (biases stay 0).
static Model *init_model(Arena *arena) {
    Model *m = arena_make_type(arena, 1, Model);
    memset(m, 0, sizeof(Model));
    he_fill(m->W1, H1 * IN, IN, 1);
    he_fill(m->W2, H2 * H1, H1, 2);
    he_fill(m->W3, OUT * H2, H2, 3);
    return m;
}

// ReLU activation: relu(x) = max(0, x).
static f32 relu(f32 x) {
    return x > 0.0f ? x : 0.0f;
}

// ReLU derivative: relu'(x) = 1 if x > 0 else 0.
static f32 relu_grad(f32 x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Dense layer: out = W * in + b, with W stored row-major as [n_out][n_in].
static void dense_forward(f32 *in, f32 *W, f32 *b, f32 *out, i32 n_in, i32 n_out) {
    for (i32 o = 0; o < n_out; o++) {
        f32 s = b[o];
        f32 *row = W + (u64)o * n_in;
        for (i32 i = 0; i < n_in; i++) s += row[i] * in[i];
        out[o] = s;
    }
}

// Numerically stable softmax: p_i = e^(z_i - max z) / sum_j e^(z_j - max z).
static void softmax(f32 *z, f32 *p, i32 n) {
    f32 mx = z[0], sum = 0.0f;
    for (i32 i = 1; i < n; i++)
        if (z[i] > mx) mx = z[i];
    for (i32 i = 0; i < n; i++) {
        p[i] = expf(z[i] - mx);
        sum += p[i];
    }
    for (i32 i = 0; i < n; i++) p[i] /= sum;
}

// Forward pass: a1 = relu(W1 x + b1), a2 = relu(W2 a1 + b2), p = softmax(W3 a2 + b3).
static Cache forward(Model *m, f32 *x) {
    Cache c;
    dense_forward(x, m->W1, m->b1, c.z1, IN, H1);
    for (i32 i = 0; i < H1; i++) c.a1[i] = relu(c.z1[i]);
    dense_forward(c.a1, m->W2, m->b2, c.z2, H1, H2);
    for (i32 i = 0; i < H2; i++) c.a2[i] = relu(c.z2[i]);
    dense_forward(c.a2, m->W3, m->b3, c.z3, H2, OUT);
    softmax(c.z3, c.p, OUT);
    return c;
}

// Cross-entropy loss for one sample: L = -log(p_y).
static f32 cross_entropy(f32 *p, i32 label) {
    ASSERT_MSG(label >= 0 && label < OUT, "label %d out of range", (int)label);
    return -logf(p[label] + 1e-12f);
}

// Backprop: d3 = p - y; d2 = (W3^T d3) * relu'(z2); d1 = (W2^T d2) * relu'(z1); dW = d * a^T, db = d (accumulated into g).
static void backward(Model *m, f32 *x, i32 label, Cache c, Model *g) {
    f32 d1[H1], d2[H2], d3[OUT];
    for (i32 i = 0; i < OUT; i++) d3[i] = c.p[i] - (i == label ? 1.0f : 0.0f);
    for (i32 i = 0; i < OUT; i++) {
        for (i32 j = 0; j < H2; j++) g->W3[i * H2 + j] += d3[i] * c.a2[j];
        g->b3[i] += d3[i];
    }
    for (i32 j = 0; j < H2; j++) {
        f32 s = 0.0f;
        for (i32 i = 0; i < OUT; i++) s += m->W3[i * H2 + j] * d3[i];
        d2[j] = s * relu_grad(c.z2[j]);
    }
    for (i32 j = 0; j < H2; j++) {
        for (i32 k = 0; k < H1; k++) g->W2[j * H1 + k] += d2[j] * c.a1[k];
        g->b2[j] += d2[j];
    }
    for (i32 k = 0; k < H1; k++) {
        f32 s = 0.0f;
        for (i32 j = 0; j < H2; j++) s += m->W2[j * H1 + k] * d2[j];
        d1[k] = s * relu_grad(c.z1[k]);
    }
    for (i32 k = 0; k < H1; k++) {
        for (i32 i = 0; i < IN; i++) g->W1[k * IN + i] += d1[k] * x[i];
        g->b1[k] += d1[k];
    }
}

// SGD with momentum: v = mu * v - lr * g / batch; w = w + v; then clears the gradients.
static void sgd_step(Model *m, Model *g, Model *v, f32 lr, f32 momentum, i32 batch) {
    ASSERT(batch > 0);
    f32 *w = (f32 *)m, *gr = (f32 *)g, *ve = (f32 *)v;
    for (i32 i = 0; i < NPARAMS; i++) {
        ve[i] = momentum * ve[i] - lr * gr[i] / (f32)batch;
        w[i] += ve[i];
        gr[i] = 0.0f;
    }
}

// Computes accuracy = correct / N and the mean cross-entropy loss over a dataset.
static Eval evaluate(Model *m, Dataset d) {
    i32 correct = 0;
    f64 loss = 0;
    for (i32 i = 0; i < d.n; i++) {
        Cache c = forward(m, d.x + (u64)i * IN);
        i32 best = 0;
        for (i32 k = 1; k < OUT; k++)
            if (c.p[k] > c.p[best]) best = k;
        if (best == d.y[i]) correct++;
        loss += cross_entropy(c.p, d.y[i]);
    }
    if (d.n == 0) return (Eval){0};
    return (Eval){(f32)correct / (f32)d.n, (f32)(loss / d.n)};
}

// Trains with shuffled minibatch SGD and keeps the weights with the best validation accuracy.
static void train(Arena *arena, Model *m, Dataset tr, Dataset va, i32 epochs) {
    const i32 batch = 32;
    f32 lr = 0.01f, momentum = 0.9f, best_acc = -1.0f;
    Model *g = arena_make_type(arena, 1, Model);
    Model *v = arena_make_type(arena, 1, Model);
    Model *best = arena_make_type(arena, 1, Model);
    memset(g, 0, sizeof(Model));
    memset(v, 0, sizeof(Model));
    i32 *perm = arena_make_type(arena, tr.n, i32);
    for (i32 i = 0; i < tr.n; i++) perm[i] = i;

    for (i32 e = 0; e < epochs; e++) {
        for (i32 i = tr.n - 1; i > 0; i--) {
            i32 j = (i32)random_range_u32(0, (u32)i);
            i32 t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        f64 loss = 0;
        for (i32 s = 0; s < tr.n; s += batch) {
            i32 end = s + batch < tr.n ? s + batch : tr.n;
            for (i32 i = s; i < end; i++) {
                f32 *x = tr.x + (u64)perm[i] * IN;
                Cache c = forward(m, x);
                loss += cross_entropy(c.p, tr.y[perm[i]]);
                backward(m, x, tr.y[perm[i]], c, g);
            }
            sgd_step(m, g, v, lr, momentum, end - s);
        }
        Eval ev = evaluate(m, va);
        if (ev.acc > best_acc) {
            best_acc = ev.acc;
            memcpy(best, m, sizeof(Model));
        }
        if (e % 5 == 0 || e == epochs - 1) {
            ArenaMark mark = arena_mark(arena);
            println_string(string_format(arena, "epoch %3d  train loss %.4f  val loss %.4f  val acc %.1f%%",
                                         (int)(e + 1), loss / tr.n, ev.loss, ev.acc * 100.0f));
            arena_pop_to(arena, mark);
        }
        lr *= 0.97f;
    }
    memcpy(m, best, sizeof(Model));
    println_string(string_format(arena, "Best validation accuracy: %.1f%%", best_acc * 100.0f));
}

// Writes magic, layer sizes, the model struct, and newline-separated class names to a binary file.
static bool save_model(Model *m, Array_Bird birds, char *path) {
    StringBuilder sb;
    sb_init(&sb, 1024);
    i32 dims[4] = {IN, H1, H2, OUT};
    sb_push_string(&sb, string_from_cstr("BIRDNN01"));
    sb_push_string(&sb, (String){.data = (u8 *)dims, .count = sizeof dims});
    sb_push_string(&sb, (String){.data = (u8 *)m, .count = sizeof(Model)});
    for (u64 i = 0; i < birds.count; i++) {
        if (i > 0) sb_push(&sb, '\n');
        sb_push_string(&sb, birds.data[i].name);
    }
    bool ok = file_write_all(path, sb_build(&sb));
    sb_free(&sb);
    return ok;
}

// Reads a model file written by save_model, validating its header and layer sizes (model is NULL on failure).
static Loaded load_model(Arena *arena, char *path) {
    String file;
    u64 head = 8 + 4 * sizeof(i32);
    if (!file_read_all(arena, path, &file)) return (Loaded){0};
    if (file.count < head + sizeof(Model) || memcmp(file.data, "BIRDNN01", 8) != 0) return (Loaded){0};
    i32 dims[4];
    memcpy(dims, file.data + 8, sizeof dims);
    if (dims[0] != IN || dims[1] != H1 || dims[2] != H2 || dims[3] != OUT) return (Loaded){0};

    Model *m = arena_make_type(arena, 1, Model);
    memcpy(m, file.data + head, sizeof(Model));
    String rest = {.data = file.data + head + sizeof(Model), .count = file.count - head - sizeof(Model)};
    Loaded ld = {m, string_split(arena, rest, '\n')};
    if (ld.names.count != OUT) return (Loaded){0};
    return ld;
}

// Classifies an audio file by averaging softmax probabilities over its segments and printing the top 3.
static void infer(Arena *arena, Mel mel, Loaded ld, char *path) {
    ArenaMark mark = arena_mark(arena);
    Audio a = load_audio(arena, path);
    i32 segs = count_segments(a.n);
    if (a.n == 0) {
        println_string(string_format(arena, "Could not decode %s (expected .mp3 or 16-bit PCM .wav)", path));
        arena_pop_to(arena, mark);
        return;
    }
    if (segs == 0) {
        println_string(string_from_cstr("Clip is too short (need at least 0.75 seconds)."));
        arena_pop_to(arena, mark);
        return;
    }
    f32 avg[OUT] = {0};
    for (i32 g = 0; g < segs; g++) {
        f32 *feat = segment_features(arena, mel, a, g);
        apply_normalization(ld.model, feat);
        Cache c = forward(ld.model, feat);
        for (i32 k = 0; k < OUT; k++) avg[k] += c.p[k] / (f32)segs;
    }
    println_string(string_format(arena, "Top predictions for %s (%d segments):", path, (int)segs));
    bool used[OUT] = {false};
    for (i32 r = 0; r < 3; r++) {
        i32 best = -1;
        for (i32 k = 0; k < OUT; k++)
            if (!used[k] && (best < 0 || avg[k] > avg[best])) best = k;
        used[best] = true;
        println_string(string_format(arena, "  %d. %-22s %5.1f%%", (int)(r + 1),
                                     cstr(arena, ld.names.data[best]), avg[best] * 100.0f));
    }
    arena_pop_to(arena, mark);
}

// Program entry: dispatches the download, train, and infer subcommands.
int main(int argc, char **argv) {
    if (argc < 2) {
        print("Usage: birds download | train | infer <file.mp3|file.wav>\n");
        return 1;
    }
    Arena arena = arena_create(MB(256));
    random_seed(SEED);
    Array_Bird birds = make_birds(&arena);
    Mel mel = build_mel_filterbank(&arena);

    if (strcmp(argv[1], "download") == 0) return download_all_data(&arena, birds) ? 0 : 1;

    if (strcmp(argv[1], "train") == 0) {
        Dataset all = build_dataset(&arena, mel, birds);
        if (all.n == 0) {
            print("No data found. Run `birds download` first.\n");
            return 1;
        }
        Split sp = split_dataset(&arena, all);
        print("Train: % segments  Validation: % segments  Test: % segmnets\n", fmt(int, sp.train.n), fmt(int, sp.val.n), fmt(int, sp.test.n));
        if (sp.train.n == 0 || sp.val.n == 0 || sp.test.n == 0) {
            print("Not enough data to split; download more clips.\n");
            return 1;
        }
        Model *m = init_model(&arena);
        normalize_dataset(m, sp.train, sp.val, sp.test);
        TIME_BLOCK("training") {
            train(&arena, m, sp.train, sp.val, 60);
        }
        Eval te = evaluate(m, sp.test);
        println_string(string_format(&arena, "Test accuracy: %.1f%% (loss %.4f, %d segments)", 
            te.acc * 100.0f, te.loss, (int)sp.test.n));
        if (!save_model(m, birds, MODEL_PATH)) {
            print("Could not write the model file.\n");
            return 1;
        }
        print("Saved model to " MODEL_PATH "\n");
        return 0;
    }

    if (strcmp(argv[1], "infer") == 0 && argc >= 3) {
        Loaded ld = load_model(&arena, MODEL_PATH);
        if (!ld.model) {
            print("Could not load " MODEL_PATH ". Run `birds train` first.\n");
            return 1;
        }
        infer(&arena, mel, ld, argv[2]);
        return 0;
    }

    print("Usage: birds download | train | infer <file.mp3|file.wav>\n");
    return 1;
}
