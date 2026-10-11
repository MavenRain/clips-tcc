/* Core ML through the Objective-C runtime: dlopen libobjc, Foundation and CoreML,
 * no -framework flags (SPEC section 6, probe/PROBE.md). Must run unsandboxed. */
#ifndef CLIPS_COREML_H
#define CLIPS_COREML_H

#include <stdint.h>

enum { CML_BATCH = 16, CML_DISC = 5 };

typedef struct cml_model cml_model;

/* Bind the runtime once. Returns 0 on success. */
int cml_init(void);

/* Load one function ("select" or "score") of an .mlmodelc. units: 0 cpu, 1 cpu+gpu, 2 all, 3 cpu+ane.
 * The input MLMultiArrays are allocated once here, with the dtype the model declares, and reused by
 * every cml_predict call. NULL on failure, with the reason on stderr. */
cml_model *cml_load(const char *path, const char *function, long units);

/* Dim 1 of the input "ids" from the model description, or fallback when it is unknown. */
long cml_width(const cml_model *m, long fallback);

/* Run one batch of CML_BATCH rows.
 * ids, mask: CML_BATCH * cml_width(m) int32, row-major, zero-filled past the tokens.
 * disc: CML_BATCH * CML_DISC float32, or NULL when the function has no "disc" input.
 * outs[k] receives CML_BATCH floats for the output names[k] (float16 and double widened to float).
 * Returns 0 on success. */
int cml_predict(cml_model *m, const int32_t *ids, const int32_t *mask, const float *disc,
                const char *const *names, int n_out, float *const *outs);

void cml_free(cml_model *m);

#endif
