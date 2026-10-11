# Session 1 stage work orders (stages 1-4 run in parallel; stage 5 after)

## Shared rules

Tool: `clips VIDEO [--limit N] [--json]` prints ranked, non-overlapping highlight spans of a video with the Desert Ant Labs Clips Core ML model. Plain C99 built with TinyCC (`tcc`, /Users/oobi/.local/bin/tcc, 0.9.28rc mob, AArch64 Darwin).

The `check-words` and `check-e2e` gates generate the ignored `fixtures/talk.mp4` from the tracked `fixtures/talk.txt` when needed, using macOS `say` (Samantha) and the ffmpeg-full binary.

- Repo /Users/oobi/Documents/clips-tcc. Reference Swift clones (read-only): SDK = /private/tmp/claude-501/-Users-oobi-Documents-claude7/826bec18-f8b0-4f21-9f9b-93a04c2dca1d/scratchpad/desert-ant-core, CLI = same dir /desert-ant-cli. If they are gone, re-clone from github.com/Desert-Ant-Labs with the sandbox off (the sandbox blocks /etc/ssl/cert.pem).
- docs/SPEC.md: read ONLY the lines your stage names, with Read offset/limit (whole-file reads over 150 lines are refused).
- Fixed interfaces: src/coreml.h, tokenizer.h, audio.h, words.h, features.h, pipeline.h, uni.h. Do not change existing declarations; add one to the header you own only if unavoidable, and report it. src/uni.c is done (uni_is_space, uni_graphemes, uni_last_grapheme, uni_trim over utf8proc); use it, never copy it.
- Makefile is fixed. It includes mk/*.mk; its header gives the fragment contract (OBJS +=, LIBS +=, CFLAGS_<module> =, CHECKS += check-<stage>; a $(BUILD)/%.o pattern rule exists; uni.o, -lutf8proc and UTF8PROC are already set). Edit ONLY your own fragment. Other stages are built IN PARALLEL by other agents: never touch their files.
- Tool-call budget: a hook cuts a subagent off at about 16 tool calls. Write your sources within your first 3-4 calls, then iterate. Batch independent calls in one message. Keep output short (tail/head).
- Shell: `rg` not grep, `sd` not sed (hook-denied). No `cd X &&`, no leading `VAR=...`; absolute paths (`make -C /Users/oobi/Documents/clips-tcc check-<stage>`). Temp files under $TMPDIR. Any run under 4 GB of memory.
- Never git add, commit or push. Do not touch models/ or fixtures/.
- Tests are C (tests/test_<stage>.c), built and run by the check target from the repo root. No shell or Python harness.
- C style: match probe/coreml_probe.c (short comments, small static helpers, early returns, errors to stderr with a return code). Free what you allocate on every path. If a hook denies a code pattern, rewrite as its message says.
- Final message under 250 words: files written; the exact gate command and its last output lines; anything unsettled (with SPEC section); any header additions.

## Stage 1: src/coreml.c (coreml.h), tests/test_coreml.c, mk/coreml.mk

Fragment: OBJS += $(BUILD)/coreml.o; CHECKS += check-coreml; check-coreml builds $(BUILD)/test_coreml from the test + $(OBJS) $(LIBS) and runs it.

Start from probe/coreml_probe.c (working dlopen of libobjc/Foundation/CoreML, objc_msgSend cast to the exact prototype per call site, half_to_float, strides) and probe/PROBE.md (select: ids/mask Int32 16x128 + disc Float32 16x5 -> saliency/start_p/end_p Float16 [16]; score: ids/mask Int32 16x256 -> score Float16 [16]). Read SPEC.md lines 166-173. Read the SDK's Sources/Inference/CoreMLSession.swift from line 200 to the end and mirror its dtype conversion for every declared input dtype (Int32, Float16, Float32, Double) and output dtype, honouring strides on both sides.

- cml_load: MLModelConfiguration functionName + computeUnits; modelWithContentsOfURL:configuration:error:. Read the description once: declared input names, each input's multiArrayConstraint dataType and shape (width = shape[1] of "ids"). Allocate one MLMultiArray per declared input ([CML_BATCH, width] for ids/mask, [CML_BATCH, CML_DISC] for disc) with the declared dtype; keep them and the model retained for reuse. Release all in cml_free.
- cml_predict: fill the reused arrays from the host int32/float32 buffers (convert, by strides), build the provider, predict, read each named output (any dtype, by strides) into outs[k][r], r < CML_BATCH. Wrap each call in objc_autoreleasePoolPush/Pop. Return -1 with a message on any nil. disc NULL with a declared "disc" = error; disc given but not declared = ignored.

Gate, run with dangerouslyDisableSandbox: true (Core ML writes ~/Library/Caches; sandboxed loads fail with a misleading "functionName property must be nil"). Cold first load ~20 s per function, warm 0.6 s.
1. cml_init; load "select" and "score" from models/clips.mlmodelc, units 2; cml_width 128 and 256.
2. Every row = [0, 35378, 8999, 38, 2], rest id 0 mask 0. select, disc all 0: saliency 0.71680, start_p 0.95752, end_p 0.07666 (within 2e-3, all 16 rows). disc {0.5, 1, 0, 1, 0.1}: 0.75439, 0.91113, 0.65186. score: 0.03741.
3. Reuse: 3 select predicts on one cml_model alternating the disc variants; values hold.
4. Short batch (row 0 only, rows 1-15 all zero): print row 0 saliency; a difference over 2e-3 is reported, not a failure.

## Stage 2: src/tokenizer.c (tokenizer.h), tests/test_tokenizer.c, mk/tokenizer.mk

Fragment: OBJS += $(BUILD)/tokenizer.o; CFLAGS_tokenizer = -I$(UTF8PROC)/include; CHECKS += check-tokenizer; check-tokenizer builds $(BUILD)/test_tokenizer from the test + $(BUILD)/tokenizer.o $(BUILD)/uni.o $(LIBS).

Read SPEC.md lines 16-73 (full algorithm with cites). Read the SDK's Sources/Clips/Tokenizer.swift and Tests/ClipsTests/ClipTests.swift lines 150-260 (vectors, syntheticVocab()/vocabBytes builders). Find what `text.nfkc` does (Sources/TextNormalization) and whether it can differ from utf8proc NFKC; report it (SPEC open question 2).

- NFKC via utf8proc_map(UTF8PROC_NULLTERM|UTF8PROC_STABLE|UTF8PROC_COMPOSE|UTF8PROC_COMPAT). Only U+0020 is a space for the squeeze.
- Exact-bytes hash index (open addressing, any hash), maxLen capped at 32, unkPenalty = (double)min(scores) - 10, Viterbi in double, strict >, pieces before unk (SPEC 1.3). Every reject rule of SPEC 1.1, including duplicate piece bytes.
- tok_encode: per-call heap scratch sized to the input (up to several thousand scalars); no global state; tok read-only after load.

Gate:
1. All SPEC 1.5 vectors, rebuilding the synthetic container in C exactly as the Swift builders do (rejects; U+0CCB vs U+0CCA U+0CD5 -> [0, 4, 2]).
2. models/clip_tokenizer.bin loads (count 250002, unk 3, bos 0, eos 2). encode("Hello world!", 128) == [0, 35378, 8999, 38, 2] (confirm against pieces "▁Hello", "▁world", "!"; fix the expectation if the vocab disagrees, and say so). encode("ೋ", 128) on the real vocab: expected id from ClipTests.swift ~:194-209 (9440 vs 80321).
3. 300-word sentence at max_len 64 -> 64 ids, first 0, last 2; max_len 0 -> 0; "" -> [0, id("▁"), 2]; "  a   b  " == "a b".
4. Each line of fixtures/talk.txt encoded 100 times in under 2 s (print the time).

## Stage 3: src/audio.c (audio.h), src/words.c (words.h), tests/test_words.c, mk/audio.mk

Fragment: OBJS += $(BUILD)/audio.o $(BUILD)/words.o; CFLAGS_audio, CFLAGS_words; LIBS += libav + whisper flags; CHECKS += check-words; check-words builds $(BUILD)/test_words from the test + $(BUILD)/audio.o $(BUILD)/words.o $(BUILD)/uni.o $(LIBS).

Libraries (USER rulings): libav from /opt/homebrew/opt/ffmpeg-full (9.0.2; include/, lib/ dylibs). NOT the Homebrew `ffmpeg` formula (8.1.2, broken). whisper.cpp from /opt/homebrew/opt/whisper-cpp (libwhisper 1.9.5, include/whisper.h; its ggml headers likely under /opt/homebrew/opt/ggml/include). Model models/ggml-base.en.bin. tcc links Homebrew dylibs with -L/-l (verified with utf8proc). If tcc cannot parse a header, report the exact error before working around it.

Read SPEC.md lines 175-203. Read the SDK's Sources/Transcript/TimedWord.swift and Sentence.swift, and find how desert-ant-cli turns ASR output into TimedWords (`rg -n TimedWord` over both Sources trees): granularity, punctuation, leading spaces (SPEC open question 8).

- audio_load: avformat_open_input, best audio stream, send/receive decode with flush, swresample to 16000 Hz mono AV_SAMPLE_FMT_FLT via AVChannelLayout, flush the resampler, geometric buffer growth, -1 with a message on failure, av_log_set_level(AV_LOG_ERROR).
- words_transcribe: whisper_init_from_file_with_params; whisper_full_default_params(WHISPER_SAMPLING_GREEDY), language "en", token_timestamps = true, print_* off, n_threads 4; whisper_log_set no-op. Tokens -> TimedWords: skip ids >= whisper_token_eot(ctx); a token whose text starts with a space starts a new word, others append; start = first t0/100, end = last t1/100; text exact, leading space kept.
- words_sentences: SPEC 7.2 exactly (runOnLimit 320; . ! ? U+2026 as the last grapheme cluster; a lone terminator does not close; untimed pending dropped; whitespace trimming via uni.h).

Gate:
1. Splitter unit cases on hand-built cw_word arrays: two sentences; lone "." not closing; leading whitespace-only words skipped; untimed run dropped; 320-grapheme run-on closes; U+2026 closes.
2. audio_load(fixtures/talk.mp4): samples/16000 within 0.5 s of the container duration (~94 s; print both).
3. Transcribe + split: print word count, sentence count, first 3 sentences with times; assert >= 3 sentences, start <= end, starts non-decreasing, last end <= duration + 0.5; print fixtures/talk.txt next to it (need not match).
Report peak memory (/usr/bin/time -l). If whisper fails sandboxed (permission/Metal), rerun that call with dangerouslyDisableSandbox: true and say so.

## Stage 4: src/features.c (features.h), src/pipeline.c (pipeline.h), tests/test_pipeline.c, mk/pipeline.mk

Fragment: OBJS += $(BUILD)/features.o $(BUILD)/pipeline.o; CFLAGS_features = -I$(UTF8PROC)/include; CHECKS += check-pipeline; check-pipeline builds $(BUILD)/test_pipeline from the test + $(BUILD)/features.o $(BUILD)/pipeline.o $(BUILD)/uni.o $(LIBS) (stub cml_*/tok_* in the test file if pipeline.o needs them).

Read SPEC.md lines 75-165 (sections 2-5; cites Sources/Clips/Pipeline.swift P and Model.swift M); confirm against those files. Settle SPEC open question 1: find the Apple backend of the SDK's Regex module (Sources/Regex) and its `\b` and case-insensitive semantics. For NSRegularExpression without useUnicodeWordBoundaries, `\b` is ICU default: word char = \p{Alphabetic}, \p{Mark}, \p{Decimal_Number}, \p{Connector_Punctuation}, U+200C, U+200D. Implement what that backend does; report it. ASCII apostrophe only, no U+2019 fold (SPEC open question 7).

- features.c: no regex library; hand-code the 16 patterns (10 hook, 6 payoff) as literal matchers with \b checks over the lowercased text, 0 or 1 each. Lowercase per scalar with utf8proc_tolower (comment: Swift lowercased() also applies SpecialCasing). ends_with_question via uni_trim. digit_ratio: clusters whose first scalar has a Numeric_Type (utf8proc Nd, Nl, No; comment the Unihan gap) / (graphemes + 1) on the lowercased text.
- pipeline.c: budget curve; anchors by STABLE saliency-descending sort (index tie-break); candidates with set dedupe, ascending (lo, hi); SPEC 5.1 DP exactly (eps 1e-6, use wins only when > skip + eps, smallest k reaching best - eps, backtrack); SPEC 5.2 Jaccard (" "-split non-empty tokens, strict > 0.5, earlier wins); SPEC 5.3 stable score sort and fields. pipeline_run: guard n < 3; selector batches of 16 (tok_encode at min(64, W), W = cml_width(select, 128), mask 1 on written tokens, disc from features_disc, rows past a short batch all zero, request saliency/start_p/end_p, use saliency); scorer batches of 16 over candidates (W2 = cml_width(score, 128), text = sentences lo..hi joined with " ", output "score"); then select + fields. It cannot run in this gate; make it compile.

Gate:
1. features: hook/payoff counts on "Here's the secret: nobody knows why." and "So that's why the result matters, which means everything." (expected counts worked out by hand); "whyever" no, "why?" yes; "the mistakes" yes (no \b in that pattern); ends_with_question with trailing spaces; digit_ratio("abc 12") = 2/7.
2. budget: limit 3 -> 3; limit 0 -> curve at 749/750, 1499/1500, 4499/4500 words.
3. candidates: n = 20, known saliency: count, order, no duplicates, each span 3..10 long containing an anchor.
4. select: DP hand cases (long high span vs two disjoint smaller ones; all-negative -> none; k cap; spans sharing a sentence never both chosen) and the Jaccard cut (later near-duplicate with repeated sentence text dropped).

## Stage 5 (after 1-4): src/main.c, mk/main.mk, end-to-end gate

Read the CLI ClipPipeline.swift:95-145 and IO/Output.swift (desert-ant-cli/Sources/DesertAntCLI) for how Pick.id/start/end/durationSec derive from the clip ranges and the --json shape; SPEC.md lines 196-206 for the ranges (padding 0.15, merge, pauses > 0.3 s cut). main: args, audio_load -> words_transcribe -> words_sentences -> cml_init/cml_load x2 -> tok_load -> pipeline_run -> ranges -> text or JSON. Gate: fixtures/talk.mp4 (94 s) gives at most 8 spans, sorted best first, no overlaps; run unsandboxed. Oracle `da clips fixtures/talk.mp4 --json` = USER ruling later.
