# Core ML probe (2026-10-11)

VERDICT: works. A plain C file built by TinyCC loads clips.mlmodelc and runs both functions through the Objective-C runtime.

## Build and run

    ./probe/build.sh
    probe/build/coreml_probe models/clips.mlmodelc 2

The build script creates `probe/build` before invoking TinyCC, including on a fresh checkout.

- No `-framework` and no `-lobjc`. The probe dlopens /usr/lib/libobjc.A.dylib, Foundation and CoreML, then dlsyms objc_getClass, sel_registerName, objc_msgSend, objc_autoreleasePoolPush/Pop.
- objc_msgSend is cast to the exact prototype at each call site.
- MLModelConfiguration.functionName = "select" or "score". computeUnits 2 (All).

## Real I/O

| function | inputs | outputs |
|---|---|---|
| select | ids Int32 16x128, mask Int32 16x128, disc Float32 16x5 | saliency, start_p, end_p: Float16 [16], stride 1 |
| score | ids Int32 16x256, mask Int32 16x256 | score: Float16 [16], stride 1 |

Input strides are contiguous. Outputs need a Float16 to float conversion (half_to_float in the probe).

## Values (every row = [0, 35378, 8999, 38, 2], rest id 0 mask 0)

- select, disc all 0: saliency 0.71680, start_p 0.95752, end_p 0.07666.
- select, disc {0.5, 1, 0, 1, 0.1}: saliency 0.75439, start_p 0.91113, end_p 0.65186.
- score: 0.03741.
- All 16 rows are finite and equal. The disc input changes the output, so it is wired.

## Timings (M-series, units All)

- First load of a function: 21-23 s (device compile). Later loads: 0.6 s (cache).
- First predict: 0.9-3.9 s. Warm predict: 104-171 ms per batch of 16.

## Sandbox

The tool must run outside the Claude Code sandbox. Core ML writes a cache to ~/Library/Caches/<process name>. In the sandbox that write fails, and the load then reports a misleading error: "functionName property must be nil unless the model type is ML Program".
