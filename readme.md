<div align="center">

# Bird Classifier

A fully from scratch bird classifier in C.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Roadmap](#roadmap)
- [Notes](#notes)
- [License](#license)

---

## Overview

Bird sound classifier: a small fully-connected neural network written from scratch in C.

Species (10 common birds around Bethesda, MD): American Robin, Northern Cardinal,
Carolina Chickadee, Tufted Titmouse, Blue Jay, Carolina Wren, Mourning Dove,
American Crow, Northern Mockingbird, House Sparrow.

Architecture: 512 inputs -> 128 (ReLU) -> 64 (ReLU) -> one softmax output per target bird
plus a final "No target bird" class, trained with
cross-entropy loss, minibatch SGD with momentum, and He-initialized weights.
The 512 inputs are a 32 mel-band x 16 time-bin log-mel spectrogram of a 3 second segment.

Data: `download` asks the Xeno-canto API v3 (via the curl command, needs XC_API_KEY set)
for quality-A recordings of each species, parses the JSON reply with cJSON, and downloads
each clip into DATA_DIR/mp3/. A manifest (DATA_DIR/manifest.txt) lists "<species epithet> (<name>) <file name>"
per clip. Every data location derives from the single DATA_DIR define, so the whole data folder can be moved
by changing that one line.
Each MP3 is decoded in-process with minimp3, mixed to mono, and resampled to 16 kHz.
It is cut into up to four 3 second segments, each run through a Hann-windowed 512-point
FFT, a triangular mel filterbank, log(1 + x), and pooled in time to the 32x16 grid.
The "No target bird" class is trained on the audio files listed (one path per line) in
DATA_DIR/none.txt, such as wind, traffic, speech, or non-target species. If none are supplied,
training prints a notice and proceeds with that class never appearing as a target.
At inference, a prediction of that class, or a top probability below MIN_CONFIDENCE, is
reported as "likely none of the target birds".
Clips (not segments) are split 60/20/20 into train/validation/test so no clip leaks across sets.

Memory: everything allocates from one base-library Arena; temporary work is rolled back
with arena marks. Text handling, parsing, file IO, hash maps, arrays, and randomness all
come from the base library.

## Installation

### Steps

Setup:
```bash
source .envrc
```

Build:
```bash
make
```
Use:
```bash
make run
```
or
```bash
./birds download
```

## Roadmap

## Notes
- data is stored in data/

## License

Distributed under the GNU License. See [`LICENSE`](LICENSE) for details.
