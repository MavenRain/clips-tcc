# Clips highlight pipeline: C-portable spec

Target: plain C under TinyCC. Inference through Core ML via the Objective-C runtime. Transcript words from whisper.cpp word timestamps. Voz is out of scope.

Sources: SDK = desert-ant-core, CLI = desert-ant-cli (local clones under the scratchpad). Citations are `file:line` in those trees.

## 0. Known facts (from the previous agent; each one is verified or corrected in the sections below)

1. Rows are zero-filled: unused positions hold id 0 with mask 0, not pad id 1 (Model.swift:94-95,126-127). Selector cuts each sentence at 64 tokens inside its 128-wide row (Model.swift:40,99,157). Truncation keeps eos: first max-1 ids plus eos (Tokenizer.swift:108-109).
2. RDTK version byte skipped (Tokenizer.swift:28); the hash table is built in memory, not stored. Only U+0020 maps to U+2581; U+2581 always prefixed; empty sentence = [0, "▁", 2] (Tokenizer.swift:118-138). Viterbi uses strict `>` with longest piece first; each unknown scalar gets its own unk id with penalty min(scores)-10; scores summed as doubles (Tokenizer.swift:92,149-169).
3. start_p/end_p are fetched and discarded (Model.swift:86-87). Anchors = top max(budget*4, 12) by saliency. No activation on any output.
4. Near-duplicate cut (Jaccard 0.5 over word sets split on " ", case-sensitive) runs in time order before the score sort, so the earlier span wins (Pipeline.swift:82-85,165-171).
5. LiteRT roles only from clips_meta.json: args_0 ids int64, args_1 mask int64, args_2 disc float32.
6. CLI passes limit nil (duration curve 8/11/13/14). Pick.id 1-based. Sentence end padded by min(0.15, gap/2), ranges merged, pauses > 0.3 s cut (Clip.swift:92-117). Discourse features need Unicode lowercase, ICU `\b`, Character.isNumber, grapheme counts.

## 1. Tokenizer (XLM-R SentencePiece Unigram over `clip_tokenizer.bin`)

All refs are `desert-ant-core/Sources/Clips/Tokenizer.swift` unless named.

### 1.1 RDTK container (all integers little-endian)

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | magic | bytes `0x52 0x44 0x54 0x4B` ("RDTK") (:27) |
| 4 | 1 | version | NOT read; parsing starts at offset 5 (:28) |
| 5 | 4 | unk id | i32 (u32 bit pattern as Int32) (:43,:45) |
| 9 | 4 | bos id | i32 (:45) |
| 13 | 4 | eos id | i32 (:45) |
| 17 | 4 | count | i32, must be `> 0` and `<= (filesize - 21) / 6` (:46-47) |
| 21 | 4*count | scores | f32 bit patterns, one per id (:51-54) |
| 21+4c | 2*count | lengths | u16 byte length of each piece (:58-61) |
| 21+6c | rest | pieces | UTF-8 bytes back to back, id order; piece `id` = bytes[bounds[id] .. bounds[id+1]) (:66-80) |

Reject (return NULL) when: filesize < 21 or > INT32_MAX (:26); magic wrong; a length runs past the end (:70); the pieces do not end exactly at EOF (`offset == bytes.count`, :81); unk/bos/eos not in `[0, count)` (:83); two pieces have identical bytes (:210-238).

Derived at load (not stored):
- `maxLen = min(max over pieces of scalar count, 32)`, scalar count = number of bytes with `(b & 0xC0) != 0x80`; starts at 1 (:68-77, :91).
- `unkPenalty = (double)min(scores) - 10.0` (min over the f32 array, then widened) (:92).
- Hash index: open addressing, capacity = smallest power of two >= 16 and >= 2*count, linear probe `(slot+1) & mask`, empty = -1, keyed on the EXACT piece bytes (:212-243, :266-277). Hash is FNV-1a 64 (`h = 0xCBF2_9CE4_8422_2325`, `h ^= byte`, :285-287); any hash works in C as long as the lookup is exact byte equality. Do NOT key on canonically-equivalent strings: xlm-roberta-base has 29 byte-distinct, canonically-equal piece pairs and both ids must stay reachable (:186-195).

### 1.2 Normalization (`tokenize`, :114-132)

1. NFKC the input (`text.nfkc`, :115). No lowercasing (:3-4). The C port needs a full NFKC (for example utf8proc `UTF8PROC_COMPOSE|UTF8PROC_COMPAT|UTF8PROC_STABLE`); see Open questions.
2. Whitespace squeeze over scalars: only U+0020 counts as space. Leading spaces dropped (`lastWasSpace = true` at start), runs collapsed to one, one trailing space removed (:118-130). Tabs, newlines, NBSP (after NFKC NBSP becomes U+0020) are otherwise kept as ordinary scalars.
3. Prefix U+2581 and replace each remaining U+0020 by U+2581 (:131-132). Result is never empty: an empty or all-space input becomes the single scalar U+2581, so `encode("") = [bos, id("▁"), eos]` (the `n == 0` branch at :138 is unreachable).

### 1.3 Viterbi (:136-169)

```
s = scalars of normalized; n = |s|; span[i] = UTF-8 byte offset of scalar i
best[0] = 0; best[1..n] = -1e18; backPos/backID = -1         (doubles)
for i in 1..n:
  for j in max(0, i-maxLen) .. i-1 (ascending: longest piece first):
    if bytes[span[j]..span[i]) is a piece tid:
      sc = best[j] + (double)scores[tid]
      if sc > best[i]: best[i]=sc; backPos[i]=j; backID[i]=tid      (strict >)
  cand = best[i-1] + unkPenalty
  if cand > best[i]: best[i]=cand; backPos[i]=i-1; backID[i]=unkID   (strict >, after pieces)
backtrack from n via backPos, emit backID, reverse.
```
Ties keep the earlier (longer) candidate; a piece beats unk on a tie. Each unknown scalar is its own unk id (no merging of unk runs). Scores are summed in double.

### 1.4 encode(text, maxLength) (:103-110)

`ids = [bos] + tokenize(text) + [eos]`; if `maxLength <= 0` return empty; if `|ids| > maxLength` return `ids[0 .. maxLength-1) + [eos]` (keeps eos, HuggingFace `truncation=True`). Output int32.

### 1.5 Tokenizer test vectors (desert-ant-core/Tests/ClipsTests/ClipTests.swift)

- :166-173 `encode(String(repeating: "a b c ", count: 20), maxLength: 8)` -> 8 ids, first = bos, last = eos (synthetic vocab).
- :175-181 `encode("a b", maxLength: 128)` -> 4 ids `<s> ▁a ▁b </s>`.
- :183-186 `Tokenizer(bytes: [0x00, 0x01, 0x02, 0x03, 0x04])` -> nil; the synthetic vocab minus its last 4 bytes -> nil.
- :194-209 vocab pieces `["<s>", "<pad>", "</s>", "<unk>", "\u{2581}\u{0CCB}", "\u{2581}\u{0CCA}\u{0CD5}"]`, scores `[0, 0, 0, 0, -1, -1]` must load, and `encode("\u{0CCB}", maxLength: 128) == [0, 4, 2]`. In xlm-roberta-base the same pair is ids 9440 and 80321 (:196-197).
- The `syntheticVocab()` / `vocabBytes(pieces:scores:)` builders are in the same file; a C test can rebuild the container from section 1.1.

## 2. Selector (per-sentence saliency)

Refs: `Sources/Clips/Model.swift` (M) and `Sources/Clips/Pipeline.swift` (P).

- Guard: fewer than `minSentences = 3` sentences -> return no clips, no inference (M:72, P:10).
- Batch = 16 (M:24). Batches are `stride(0, n, 16)`; the last batch can be short (M:92-93).
- Per batch: `ids` int32 `[16, W]`, `mask` int32 `[16, W]`, `disc` float32 `[16, 5]`, all zero-filled (M:94-96). `W = selector input width of "ids"` from the loaded graph, fallback 128 (M:57, M:28).
- Row r: `encode(sentence, maxLength: min(64, W))`, written at `ids[r*W + c]` with `mask[r*W + c] = 1` for each written token, `c < cut` (M:98-99, M:155-161, `sentenceTokens = 64` M:40). Positions past the tokens keep id 0 and mask 0 (id 0 is `<s>` in XLM-R; pad id 1 is never written). VERIFIED fact 1.
- Rows past the end of a short batch stay all zero (ids, mask and disc); their outputs are ignored (M:144-146, M:115).
- Run with inputs `"ids"`, `"mask"`, `"disc"`; request outputs `["saliency", "start_p", "end_p"]`; fail unless 3 outputs come back and `out[0]` has `>= slice.count` float32 values; `saliency[start + r] = (double)values[r]` (M:107-115). `start_p`/`end_p` are not used (M:86-87). No activation. VERIFIED fact 3.
- Disc row r (order fixed, P:180): `[relative_position, hook_cue_count, payoff_cue_count, ends_with_question, digit_ratio]`, written at `disc[r*5 + c]` (M:103-105).

### 2.1 relative_position
`(double)(start + r) / (double)max(n, 1)`, stored as float (M:104). Range `[0, (n-1)/n]`.

### 2.2 hook_cue_count and payoff_cue_count
`t = sentence.lowercased()` (Swift full Unicode lowercase) (P:198). Each count = number of PATTERNS in the list that match anywhere in `t` (0 or 1 per pattern, not the match count) (P:181, P:199-200). Each pattern is compiled case-insensitive; a pattern that fails to compile is dropped (P:193-195).

hookPatterns (P:182-188), Swift literals verbatim:
```
"\\bhere'?s\\b", "nobody", "most people",
"the (mistake|secret|truth|problem|reason)", "\\bwhat if\\b", "\\bwhy\\b",
"\\bhow (to|i|we|you)\\b", "\\?",
"\\b(never|always|everyone|no one|nobody)\\b",
"\\b(biggest|worst|best|craziest|first|only|most)\\b",
```
payoffPatterns (P:189-191):
```
"\\bso\\b", "that'?s why", "the point", "which means", "turns out", "the result",
```
After unescaping, `\\b` is the regex `\b` and `\\?` is a literal `?`. The apostrophe is ASCII U+0027. Max hook count 10, max payoff count 6.

### 2.3 ends_with_question
`t.trimmedWhitespace.hasSuffix("?") ? 1 : 0` (P:202).

## 3. Candidates (P:35-76, M:71-80)

- Budget (P:51-53): caller `limit` given -> `max(1, limit)`; else the duration curve (P:35-41): `words = sum over sentences of max(1, number of non-empty " "-separated tokens)`; `minutes = words / 2.5 / 60` (`wordsPerSecond = 2.5`, P:20); `< 5 -> 8`, `< 10 -> 11`, `< 30 -> 13`, else `14`. The budget is computed once and used by both enumeration and ranking (M:74-78).
- Anchors (P:64-65): sentence indices sorted by saliency descending, first `max(budget * 4, 12)`. Ties: Swift `sorted` (stable in current stdlib) keeps the lower index first; a C port must use a stable sort (merge sort or index tie-break).
- Spans (P:67-72): for each anchor `a < n`, `lo` from `max(0, a - 8)` to `a`, `hi` from `a` to `min(n - 1, a + 14)`, keep when `3 <= hi - lo + 1 <= 10` (`lookback = 8`, `lookahead = 14`, `minSentences = 3`, `maxSentences = 10`, P:8-11).
- Identical spans are kept once (a set of index arrays, P:66, P:71). Output order: ascending `(lo, hi)` (P:75). No candidates -> no clips (M:79).

## 4. Scorer (per-span score, M:80, M:121-141)

- Span text: the sentences `lo..hi` joined with one U+0020 (M:80).
- Batches of 16 over the candidate list, in candidate order (M:124-125).
- `ids`, `mask` int32 `[16, W2]`, zero-filled; `W2` = scorer input width of `"ids"` from the loaded graph, fallback 128 (M:58, M:49). The 128 package serves `[16,128]` and the 256 package `[16,256]` (M:45-46).
- Row r: `encode(text, maxLength: W2)` (truncate keeps eos), mask 1 on written tokens (M:129, M:157-161).
- Inputs `"ids"`, `"mask"`; output `["score"]`; fail unless `out.first` has `>= slice.count` float32 values; `score = (double)values[r]`, raw, no activation (M:131-138).

## 5. Selection (P:79-174)

Input: `spans[i] = (lo, hi, score)` in candidate order (P:81).

### 5.1 Weighted interval scheduling, at most k = budget (P:117-159)
```
sorted = spans sorted by hi ascending, then lo ascending          (P:119; keys are unique after the set dedupe)
eps = 1e-6; if empty or budget <= 0 -> []
prev[i] = largest j < i with sorted[j].hi < sorted[i].lo, else -1   (P:123-127; sharing a sentence = overlap)
neg = -DBL_MAX
dp[0][k] = 0 for k in 0..budget; dp[i][k] = neg otherwise; take = false
for i in 1..N: p = prev[i-1] + 1
  for k in 0..budget:
    skip = dp[i-1][k]
    use  = (k >= 1 && dp[p][k-1] > neg/2) ? dp[p][k-1] + sorted[i-1].score : neg
    if use > skip + eps: dp[i][k] = use; take[i][k] = true  else dp[i][k] = skip
totals = dp[N]; best = max(totals); bestK = smallest k with totals[k] >= best - eps
backtrack: i = N, k = bestK; while i > 0 && k > 0:
  if take[i][k]: emit sorted[i-1]; k -= 1; i = prev[i-1] + 1  else i -= 1
reverse -> chosen, in ascending (hi, lo) order = time order
```
Consequence: a span with `score <= eps` (including any negative score) is never chosen, so all-negative scores give zero clips. The max is taken first, then the smallest k (P:142-152).

### 5.2 Near-duplicate cut (P:161-174), runs on `chosen` in time order (P:82-84)
For each span s in order: `A` = set of " "-split non-empty tokens over sentences `s.lo..s.hi` (raw text, case-sensitive, punctuation attached). Drop s if for any already kept span k: `|A ∩ B| / max(|A ∪ B|, 1) > 0.5` (`jaccardMax = 0.5`, P:15). Earlier span wins. VERIFIED fact 4.

### 5.3 Final order and Clip fields (P:85-96)
Stable sort of the kept spans by score descending (P:85). For rank i (0-based) of `count` spans:
- `id = i` (0-based in the SDK) (P:90)
- `sentenceIDs = [lo ... hi]` (P:91)
- `text` = those sentences joined with " " (P:92)
- `score` = raw scorer value (P:93)
- `percentile = count > 1 ? 1.0 - i / (count - 1) : 1.0` (P:94)
- `estimatedDurationSec = words / 2.5`, `words = sum of " "-split non-empty token counts` (no max(1) here, unlike the budget) (P:88, P:95)

### 2.4 digit_ratio (Pipeline.swift:203-205)
`digits = number of Characters (grapheme clusters) c in t with Character.isNumber`; `digitRatio = (float)digits / (float)(t.count + 1)`, where `t.count` is the grapheme-cluster count of the lowercased sentence. `Character.isNumber` is true when the cluster's first scalar has a Numeric_Type (so `½`, `٣`, `Ⅳ` count, not only ASCII digits).

### 2.5 trimmedWhitespace (Pipeline.swift:209-221)
Strip leading and trailing Unicode scalars with the `White_Space` property (`scalar.properties.isWhitespace`), then test the last scalar for `?`.

## 6. Core ML I/O

- Artifact: `"\(stem).mlmodelc"`, one multifunction model; functions `"select"` (selector) and `"score"` (scorer) (Catalog.swift:69-71). Set `MLModelConfiguration.functionName` before load; without it Core ML silently loads the default function, so both stages would run the selector (Catalog.swift:65-68). Needs macOS 15 / iOS 18; older OS = error, not fallback (Sources/Inference/CoreMLSession.swift:48-62; ModelLoading.swift:44-47).
- Selector inputs `"ids"` `[16, W]`, `"mask"` `[16, W]`, `"disc"` `[16, 5]`; outputs `"saliency"` `[16]`, `"start_p"` `[16]`, `"end_p"` `[16]` (Model.swift:10-11, :107-111). Scorer inputs `"ids"`, `"mask"` `[16, W2]`; output `"score"` `[16]` (Model.swift:12, :131-134).
- Host buffers are int32 (ids, mask) and float32 (disc) (Model.swift:108-110, :132-133). The session builds each `MLMultiArray` with the dtype the MODEL DECLARES for that input (`desc[name]?.multiArrayConstraint?.dataType`) and converts, including float16 ("feeds the model's native I/O precision") (CoreMLSession.swift:10, :213-214, :237-241). C port: read `MLModelDescription.inputDescriptionsByName[name].multiArrayConstraint.dataType`, convert, honour `strides` on both inputs and outputs, and widen any float16/double output to float32 before use.
- Width W/W2: `inputWidth("ids")` from the model description (CoreMLSession.swift:97), fallback 128 (Model.swift:57-58).
- Core ML enum values the C side needs (from the Core ML SDK headers, not this repo): `MLMultiArrayDataTypeDouble = 0x10040`, `Float32 = 0x10020`, `Float16 = 0x10010`, `Int32 = 0x20020`.
- ObjC calls (via `objc_msgSend`): `MLModelConfiguration new`, `setFunctionName:`, `setComputeUnits:`, `MLModel modelWithContentsOfURL:configuration:error:`, `MLMultiArray initWithShape:dataType:error:`, `MLDictionaryFeatureProvider initWithDictionary:error:`, `predictionFromFeatures:error:`, `featureValueForName:`, `multiArrayValue`, `dataPointer`, `strides`.

## 7. Transcript and time mapping (whisper.cpp words -> sentences -> seconds)

Refs: `desert-ant-core/Sources/Transcript/{TimedWord,Sentence,Clip}.swift` (TW, SE, CL). Correction: the range code cited as `Clip.swift:92-117` is the CORE file `Sources/Transcript/Clip.swift`, not a CLI file.

### 7.1 TimedWord (TW:5-37)
`{text, start, end}` in seconds; `end = max(start, end)`; text kept exactly, including leading whitespace (TW:3-7). `isTimed = isfinite(start) && isfinite(end) && start >= 0` (TW:36). whisper.cpp mapping: one TimedWord per word (or token) with its text as emitted (keep the leading space), `t0/t1` centiseconds / 100; drop special tokens; an unplaced word gets a negative time.

### 7.2 Sentence splitter (SE:53-107), `runOnLimit = 320` (SE:53)
```
pending = {text = "", start = NONE, end = 0.0}; isEmpty := (start == NONE)        (SE:73-79)
for word in words:
  if pending.isEmpty and every Character of word.text isWhitespace: skip          (SE:58)
  pending.text += word.text; if word.isTimed: { if start == NONE: start = word.start; end = word.end }   (SE:88-93)
  trimmed = pending.text with leading/trailing Character.isWhitespace removed     (SE:104-106)
  close = trimmed nonempty and (last Character in ".!?…" ? graphemes(trimmed) > 1 : graphemes(trimmed) >= 320)   (SE:81-86)
  if close: take()
take(): if trimmed nonempty and start != NONE: emit Sentence{id = sentences.count, text = trimmed, start, end = max(start, end)}; always reset pending   (SE:95-100, SE:28-33)
after the loop: take()                                                           (SE:65)
```
Terminators are exactly `.` `!` `?` `…` (U+2026) as the LAST Character; a lone terminator word does not close (SE:84-85). A pending run with no timed word is dropped (its text is lost) (SE:98). The SDK receives `sentences[i].text` as the transcript strings, index = Sentence.id.

### 7.3 Clip -> time ranges (CL:92-117), default `padding = 0.15`
For each id in `sentenceIDs` sorted (ids absent from the transcript skipped, CL:93-96):
- `before = prev exists ? s.start - prev.end : s.start`; `after = next exists ? next.start - s.end : +inf` (CL:97-98).
- `share(gap) = gap <= 0 ? 0 : (isfinite(gap) ? min(padding, gap/2) : padding)` (CL:114-116).
- range = `[max(0, s.start - share(before)), s.end + share(after)]`, end clamped `>= start` (CL:99-100, CL:17-19).
- Merge in order: if `span.start <= last.end` then `last.end = max(last.end, span.end)`, else new range (CL:103-109).
- Derived (fact 6, VERIFIED as a consequence, no 0.3 constant in code): consecutive sentences with gap g merge iff `g <= 2 * padding = 0.3 s`; longer pauses are cut.
- `duration = sum of range durations` (CL:126-127).

### 7.4 CLI output records (desert-ant-cli/Sources/DesertAntCLI/Clipping/ClipPipeline.swift)
`Pick {id: Int, start: Double, end: Double, durationSec: Double, text: String, score: Double, file: String?}` (:9-17). `ClipReport {input: String, transcript: String?, sentences: Int, materialSec, transcribeSec, selectSec, cutSec: Double, clips: [Pick]}` (:20-29). Both `Encodable`; synthesized coding uses the property names as keys and omits nil optionals. How `Pick.start/end/durationSec/id` are derived (ClipPipeline.swift:95-145) and the `--json` emitter (IO/Output.swift:61-72) were NOT read: see Open questions.

## 8. Fact check of section 0

1. VERIFIED (Model.swift:94-95, :126-127, :40, :99, :157; Tokenizer.swift:108-109).
2. VERIFIED, with additions: NFKC first (:115); maxLen capped at 32 (:91); unk is tried after pieces and wins only on strict `>` (:158-159); the `[0, ▁, 2]` ids assume XLM-R bos 0 / eos 2 from the container.
3. VERIFIED (Model.swift:86-87, :115; Pipeline.swift:64-65).
4. VERIFIED; precision: clash is strictly `> 0.5`, empty tokens are not words (Pipeline.swift:166-169).
5. NOT VERIFIED and contradicted by docs/models/clips.md:106 ("The graphs name their inputs `args_0`, `args_1`, `args_2` and their outputs `output_0` to `output_2`. ... The SDK has no mapping between the names, so the first call fails."). Out of scope for the Core ML port.
6. PARTLY VERIFIED: padding applies to BOTH ends (start by half the gap before, end by half the gap after; no neighbour after = full 0.15; first sentence uses its start time as the gap), file is core `Transcript/Clip.swift`. Unicode lowercase, Character.isNumber and grapheme counts VERIFIED (Pipeline.swift:198, :203-204); "ICU `\b`" NOT verified. `limit nil` and `Pick.id 1-based` NOT verified (CLI code unread).

## 9. Open questions

1. Regex engine: `Pattern` comes from the repo's own `Regex` module with per-platform backends (Android java.util.regex, wasm JS RegExp); the Apple backend and its `\b` and case-folding semantics were not read (Pipeline.swift:194). The C port needs one fixed definition of `\b` (ASCII `\w` vs Unicode word chars).
2. NFKC source and Unicode version: `text.nfkc` from DesertAnt TextNormalization (Tokenizer.swift:115, :12-13). The C NFKC must match it byte for byte.
3. Shipped graph dtypes: what `multiArrayConstraint.dataType` the `select`/`score` functions declare for `ids`/`mask`/`disc` and for outputs (int32 vs float16 vs float32), and the full conversion table past CoreMLSession.swift:241.
4. Scorer width 256 for this package is taken from the caller, not verified: ShippingArtifactTests.swift was not read; the code reads it from the graph (Model.swift:58, CoreMLSession.swift:97).
5. Anchor ties and NaN: Swift `sorted` stability is relied on implicitly (Pipeline.swift:64); behaviour for NaN saliency or NaN scores in the DP (Pipeline.swift:139, :151) is undefined.
6. CLI wiring unread: whether `limit` is nil, how `Pick.id` (0- or 1-based), `start`, `end`, `durationSec` derive from `Clip.ranges`, which padding the CLI passes, and the `--json` encoder settings (key strategy, pretty print, batch collection) (ClipPipeline.swift:95-145; IO/Output.swift:61-72).
7. Apostrophes: cue patterns use ASCII `'` (Pipeline.swift:183, :190); whisper.cpp can emit U+2019, which NFKC does not fold and the features do not normalize. Decide whether to keep parity with the reference (no fold).
8. Splitter quirk: `isEmpty` means "no timed word yet", so whitespace-only words are skipped only before the first timed word, and an all-untimed sentence is dropped with its text (Sentence.swift:58, :79, :98). whisper.cpp sub-word tokens versus words: confirm which granularity feeds TimedWord.
9. Compute units default and the `ComputeUnits` mapping passed to the session (ModelLoading.swift:44-47) were not read.
