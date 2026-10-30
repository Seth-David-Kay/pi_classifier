#include "base/base_all.h"
#include "small_tui/tui.h"

// Small nn to identiy bird call signals

#define N 16

structdef(Drawing){
	bool pixels[N][N];
}; structdef_end(Drawing)

Drawing paint(void);
void init_nn_files(void);
void dot(f64**, f64**, int, int, f64**);
void print_matrix(f64**, int, int);

int main(void){
    f64 a0[] = {1, 2, 3};
    f64 a1[] = {4, 5, 6};
    f64* a[] = {a0, a1};

    f64 b0[] = {7, 8};
    f64 b1[] = {9, 10};
    f64 b2[] = {11, 12};
    f64* b[] = {b0, b1, b2};

    f64 r0[2] = {0};
    f64 r1[2] = {0};
    f64* result[] = {r0, r1};

    dot(a, b, 2, 2, result);

    print_matrix(a, 2, 3);
    printf("\n");
    print_matrix(b, 3, 2);
    printf("\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++)
            printf("%f ", result[i][j]);
        printf("\n");
    }
    // init_nn_files();
    // for (int i = 0; i < 1; i++){
    // 	Drawing character = paint();
    // 	printf("int pixels[%d][%d] = {\n", N, N);
    //     for (int y = 0; y < N; y++) {
    //         printf("    {");
    //         for (int x = 0; x < N; x++)
    //             printf("%d%s", character.pixels[y][x] ? 1 : 0, x < N - 1 ? ", " : "");
    //         printf("}%s\n", y < N - 1 ? "," : "");
    //     }
    // }
}

void print_matrix(f64** matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++)
            printf("%f ", matrix[row][col]);
        printf("\n");
    }
}

void dot(f64** a, f64** b, int rows, int cols, f64** result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            f64 count = 0;
            for (int k = 0; k < cols; k++) {
                printf("a[%d][%d] b[%d][%d]\n", i,k, k,j);
                count += a[i][k] * b[k][j];
            }
            result[i][j] = count;
            printf("\n");
        }
    }
}

void init_nn_files(void) {
    f64 m[256] = {0};
    f64 w[26][256] = {{0}};
    f64 w_nonlinear[128][26] = {{0}};
    f64 b[26] = {0};
    discard (m); discard (w); discard (w_nonlinear); discard (b);
    file_write_all("m.zfx", string_from_cstr("hi1"));
}

void load_nn(Arena arena, Array_f64 x, Array_f64 w, Array_f64 w_nonlinear, Array_f64 b) {
    discard (x), discard (w), discard (w_nonlinear), discard (b);
    String text;
    f64 n;
    if (file_read_all(&arena, "/w.zxf", &text)) {
        Array_String lines = string_split(&arena, string_trim(text), '\n');
        parse_f64(lines.data[0], &n);
    }
}

Drawing paint(void) {
    Drawing drawing = {0};
    bool mode = true;
    int  last_x = -1, last_y = -1;
    uint64_t black = COLOR(C_RGB(0, 0, 0), C_RGB(0, 0, 0));
    uint64_t white = COLOR(C_RGB(5, 5, 5), C_RGB(5, 5, 5));
    uint64_t ui    = COLOR(C_BRIGHT_WHITE, C_SHADE(3), C_CYAN);
    while (tui_run(0)) {
        fill(0, 0, A, A, ui);
        scope (A, A, N * 2 + 2, N + 6) {
            panel("Mini Paint", 0, 0, ~0, ~0, ui);
            scope (1, 1, N * 2, N) {
                struct rect r = abs_xywh(0, 0, N * 2, N);   // screen position
                bool press = is_event_key(MOUSE_EVENT, LEFT_BUTTON);
                bool drag  = is_event_key(MOUSE_EVENT, LEFT_DRAG);
                if ((press || drag) && is_mouse_over(r)) {
                    int tx = (tui.event.x - r.x) / 2;
                    int ty = tui.event.y - r.y;
                    if (press) {
                        mode   = !drawing.pixels[ty][tx];
                        last_x = tx;
                        last_y = ty;
                    }
                    int x0 = last_x < 0 ? tx : last_x;
                    int y0 = last_y < 0 ? ty : last_y;
                    int steps = TUI_MAX(abs(tx - x0), abs(ty - y0));
                    for (int i = 0; i <= steps; i++) {
                        int px = steps ? x0 + (tx - x0) * i / steps : tx;
                        int py = steps ? y0 + (ty - y0) * i / steps : ty;
                        drawing.pixels[py][px] = mode;
                    }
                    last_x = tx;
                    last_y = ty;
                } else if (drag) {
                    last_x = last_y = -1;
                }

                for (int y = 0; y < N; y++)
                    for (int x = 0; x < N; x++)
                        fill(x * 2, y, 2, 1, drawing.pixels[y][x] ? white : black);
            }
            if (button("Clear", 1, ~1, A, A, ui))
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        drawing.pixels[i][j] = false;
            if (button("Quit", ~1, ~1, A, A, ui))
                tui_quit();
        }
        if (is_key_press('q'))
            tui_quit();
    }
    return drawing;
}
