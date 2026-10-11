# Session 1 kickoff: build clips-tcc

Goal: `clips VIDEO [--limit N] [--json]` prints ranked, non-overlapping highlight spans (start, end, score, text). Plain C, built with TinyCC.

Rulings (USER, 2026-10-11): Core ML through the objc runtime (probe/PROBE.md, VERDICT works); whisper.cpp for words (Homebrew libwhisper 1.9.5, models/ggml-base.en.bin); libav from /opt/homebrew/opt/ffmpeg-full (9.0.2). The Homebrew `ffmpeg` 8.1.2 binary is broken. Never commit; stage and print the message.

Read once: docs/SPEC.md (the pipeline, with file:line cites into the Swift SDK). Its sections 6 and 7 are partial: read core Sources/Clips CoreMLSession.swift after line 241, and the CLI ClipPipeline.swift:95-145 plus IO/Output.swift. Re-clone both repos outside the sandbox (the sandbox blocks /etc/ssl/cert.pem): desert-ant-core and desert-ant-cli from github.com/Desert-Ant-Labs.

Stages, each with its own gate:
1. src/coreml.c from probe/coreml_probe.c (load once per function, reuse arrays).
2. src/tokenizer.c (RDTK, NFKC, Viterbi). Gate: the SPEC section 1 test vectors.
3. src/audio.c (libav decode + swresample to 16 kHz mono float) and src/words.c (whisper token timestamps to words, then sentences per SPEC section 7).
4. src/features.c (5 disc features, cue lists verbatim) and src/pipeline.c (anchors, candidates, scorer batches, weighted interval scheduling, Jaccard cut, budget).
5. src/main.c with JSON output. Gate: fixtures/talk.mp4 (94 s) gives at most 8 spans, sorted best first, with no overlaps.

Oracle (USER ruling later): `da clips fixtures/talk.mp4 --json` from desert-ant-cli.

Runs must be unsandboxed (Core ML cache). The subagent cap is about 16 tool calls, so briefs must write files early and append to them.
