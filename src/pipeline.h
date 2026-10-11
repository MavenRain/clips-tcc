/* Selector, candidates, scorer and selection (SPEC sections 2-5). */
#ifndef CLIPS_PIPELINE_H
#define CLIPS_PIPELINE_H

#include "coreml.h"
#include "tokenizer.h"

typedef struct { int lo, hi; double score; } cp_span;

typedef struct {
    int id;                 /* rank, 0-based (SPEC 5.3) */
    int lo, hi;             /* sentence ids, inclusive */
    double score, percentile, est_duration_sec;
    char *text;             /* sentences lo..hi joined with " " */
} cp_clip;

/* Pure stages, testable without Core ML. Each returns 0 on success; out arrays are malloc'd. */
int pipeline_budget(const char *const *sentences, int n, int limit);   /* limit <= 0: duration curve */
int pipeline_candidates(const double *saliency, int n, int budget, cp_span **out, int *count);
int pipeline_select(const cp_span *spans, int count, int budget, const char *const *sentences,
                    cp_span **out, int *kept);                         /* SPEC 5.1 + 5.2, time order */

/* Full run. Fewer than 3 sentences or no candidates gives 0 clips and returns 0.
 * *clips sorted by score descending (stable). Returns 0 on success. */
int pipeline_run(cml_model *select, cml_model *score, const tok *t, const char *const *sentences, int n,
                 int limit, cp_clip **clips, int *count);

void pipeline_free(cp_clip *c, int count);

#endif
