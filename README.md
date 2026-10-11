# clips-tcc

clips-tcc is a C port of the Clips highlight pipeline of `da clips` (Desert Ant Labs). The tool reads a video or audio file, transcribes the speech and selects the best short clips. TinyCC builds it. `docs/SPEC.md` gives the full specification.

## What the tool does

1. Decode the best audio stream to 16 kHz mono float with libav.
2. Transcribe the audio to timed words with whisper.cpp. Split the words into sentences.
3. Tokenize the sentences. Run the `select` and `score` functions of the Clips Core ML model on the CPU and Neural Engine.
4. Select the clips and write them as text or as JSON.

## Dependencies

Tested on Apple silicon macOS. The tool uses Core ML, and the Makefile reads the libraries from `/opt/homebrew/opt/<package>`.

Install the Homebrew libraries:

```sh
brew install ffmpeg-full whisper-cpp ggml utf8proc
```

`ffmpeg-full` supplies libavformat, libavcodec, libswresample and libavutil. The speech recognizer was tested with `whisper-cpp` 1.9.5.

A native Darwin TinyCC build is required separately, on `PATH` as `tcc`. The validated compiler is `tcc version 0.9.28rc 2026-09-04 mob@0fb54300 (AArch64 Darwin)`, from the [TinyCC mob sources](https://repo.or.cz/tinycc.git). The [Homebrew stable tcc formula](https://formulae.brew.sh/api/formula/tcc.json) targets x86_64 and macOS 10.15 or older, so `brew install tcc` does not supply this compiler.

## Models

`scripts/fetch-models.sh` downloads the models into `models/`:

- `clips.mlmodelc`, `clip_tokenizer.bin` and `clips_meta.json` from `desert-ant-labs/clips` on Hugging Face. The Desert Ant Labs Source-Available License 1.0 applies to the Clips weights.
- `ggml-base.en.bin` from `ggerganov/whisper.cpp` on Hugging Face.

## Build and usage

```sh
scripts/fetch-models.sh
make
build/clips VIDEO [--limit N] [--json] [--models DIR]
```

- `--limit N`: the maximum number of clips. Without it, the duration curve sets the budget.
- `--json`: write the clip report as JSON. Without it, the tool writes one line for each clip: rank, start, end, score and text.
- `--models DIR`: the model directory. The default is `models`.

Run `build/clips` outside the sandbox. Core ML writes a model cache. Progress messages go to stderr.

## Gates

`make check` runs six gates. Run them from the repository root, outside the sandbox. If `fixtures/talk.mp4` is absent, the first run makes it from `fixtures/talk.txt` with `say` and ffmpeg. The E2E gate compares `materialSec` with the container duration reported independently by ffprobe.

The table records the S7 run on the original fixture. Speech synthesis and ASR can vary by machine, so regenerated fixtures can have different durations and sentence counts.

| Gate | What it checks | Result at S7 |
| --- | --- | --- |
| `check-coreml` | Model load, input widths and pinned outputs with units 3 | 14 passed, 0 failed |
| `check-tokenizer` | The tokenizer | 50 passed, 0 failed |
| `check-audio` | Decode timelines: delays, gaps, MPEG-TS and MP3 timestamps, AC3 layout copies and rate changes | 14 scenarios, 0 failed |
| `check-words` | Words and sentences of `fixtures/talk.mp4` | OK, 307 words, 38 sentences |
| `check-pipeline` | Clip selection | 53 checks, 0 failed |
| `check-e2e` | `build/clips fixtures/talk.mp4 --json` | 8 spans, 119 checks, 0 failed |

## JSON output and `da clips --json`

`--json` writes a pretty-printed `ClipReport` with sorted keys. Relative to `da clips --json`:

- Added: `sentenceIDs` in each clip. It gives the sentence ids of the clip.
- Left out: `transcript` and `Pick.file`. The tool does not cut clip files.
- `cutSec` is always 0, because the tool does not cut clips.
- `input` uses the same rule as the CLI. An absolute argument stays as given. A relative argument is appended to the working directory, and its `.` and `..` segments are removed. Symlinks are not resolved. If the working directory cannot be read, this tool keeps the argument as given.
- `transcribeSec` and `selectSec` measure each run and are not parity fields. In the selection oracle, the CLI uses a supplied transcript and reports `transcribeSec` as 0.

## Oracle result

The S3 selection oracle used desert-ant-cli `1ca0011`, which pins desert-ant-core 3.5.0 (`ec1f289`) in `Package.swift` and `Package.resolved`. The separately inspected SDK checkout `c50d20d` is 3.6.0; it was not the dependency built by that CLI.

The C tool's 38 timed sentences were supplied to the CLI through `--transcript` JSON, and a temporary CLI patch loaded this repository's `models/` directory. The comparison checks selection on the same transcript and weights. It does not compare Voz transcription with whisper.cpp.

On `fixtures/talk.mp4`, all 8 spans match in rank order. `start`, `end`, `durationSec`, `text` and `score` are exact, and `input` is equal. The structural, path and timing differences are described above.
