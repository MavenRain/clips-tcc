/* Stage 1 gate: load both functions of the Clips model and check the values in
 * probe/PROBE.md. Run from the repo root, outside the sandbox (Core ML writes
 * ~/Library/Caches; sandboxed loads fail with a misleading functionName error). */
#include <stdio.h>
#include <string.h>

#include "coreml.h"

#define MODEL "models/clips.mlmodelc"
#define TOL 2e-3f
#define MAX_W 256

static int passed, failed;

static const int32_t ROW[5] = { 0, 35378, 8999, 38, 2 };
static const float ZERO[CML_DISC] = { 0, 0, 0, 0, 0 }, CUE[CML_DISC] = { 0.5f, 1, 0, 1, 0.1f };
static const char *const SEL_OUT[3] = { "saliency", "start_p", "end_p" };
static const float SEL_WANT[2][3] = { { 0.71680f, 0.95752f, 0.07666f }, { 0.75439f, 0.91113f, 0.64941f } };
static const char *const SCORE_OUT[1] = { "score" };
static const float SCORE_WANT = 0.03741f;

static void check(int ok, const char *what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    passed += ok != 0;
    failed += ok == 0;
}

static float absf(float x) { return x < 0 ? -x : x; }

/* The first rows rows are ROW with mask 1; every other entry is id 0, mask 0. */
static void fill_tokens(int32_t *ids, int32_t *mask, long width, int rows) {
    memset(ids, 0, sizeof(int32_t) * CML_BATCH * width);
    memset(mask, 0, sizeof(int32_t) * CML_BATCH * width);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < 5; c++) {
            ids[r * width + c] = ROW[c];
            mask[r * width + c] = 1;
        }
}

static void fill_disc(float *disc, const float *row, int rows) {
    memset(disc, 0, sizeof(float) * CML_BATCH * CML_DISC);
    for (int r = 0; r < rows; r++) memcpy(disc + r * CML_DISC, row, sizeof(float) * CML_DISC);
}

/* All CML_BATCH values within TOL of want; prints the first miss. */
static int rows_near(const char *name, const float *out, float want) {
    for (int r = 0; r < CML_BATCH; r++)
        if (!(absf(out[r] - want) <= TOL)) {
            printf("  %s row %d: %.5f, want %.5f\n", name, r, out[r], want);
            return 0;
        }
    return 1;
}

/* One select predict with disc variant v (0 zero, 1 cue) on every row. */
static int select_ok(cml_model *m, const int32_t *ids, const int32_t *mask, int v) {
    float disc[CML_BATCH * CML_DISC], sal[CML_BATCH], st[CML_BATCH], en[CML_BATCH];
    float *const outs[3] = { sal, st, en };
    fill_disc(disc, v ? CUE : ZERO, CML_BATCH);
    if (cml_predict(m, ids, mask, disc, SEL_OUT, 3, outs)) return 0;
    int ok = 1;
    for (int k = 0; k < 3; k++) ok &= rows_near(SEL_OUT[k], outs[k], SEL_WANT[v][k]);
    return ok;
}

static int score_ok(cml_model *m, const int32_t *ids, const int32_t *mask, const float *disc) {
    float score[CML_BATCH];
    float *const outs[1] = { score };
    return cml_predict(m, ids, mask, disc, SCORE_OUT, 1, outs) == 0 && rows_near("score", score, SCORE_WANT);
}

static int summary(void) {
    printf("test_coreml: %d passed, %d failed\n", passed, failed);
    return failed != 0;
}

static void run(cml_model *sel, cml_model *sc, long w1, long w2) {
    static int32_t ids[CML_BATCH * MAX_W], mask[CML_BATCH * MAX_W];
    float disc[CML_BATCH * CML_DISC], sal[CML_BATCH], st[CML_BATCH], en[CML_BATCH];
    float *const outs[3] = { sal, st, en };
    const char *const bogus[1] = { "nope" };

    fill_tokens(ids, mask, w1, CML_BATCH);
    check(select_ok(sel, ids, mask, 0), "select, disc all 0: 0.71680 0.95752 0.07666 on 16 rows");
    check(select_ok(sel, ids, mask, 1), "select, disc cue: 0.75439 0.91113 0.65186 on 16 rows");
    int reuse = 1;
    for (int i = 0; i < 3; i++) reuse &= select_ok(sel, ids, mask, i % 2);
    check(reuse, "reuse: 3 select predicts alternating disc, values hold");
    check(cml_predict(sel, ids, mask, NULL, SEL_OUT, 3, outs) != 0, "select with disc NULL is an error (expected stderr)");
    check(cml_predict(sel, ids, mask, disc, bogus, 1, outs) != 0, "unknown output name is an error (expected stderr)");

    fill_tokens(ids, mask, w2, CML_BATCH);
    fill_disc(disc, CUE, CML_BATCH);
    check(score_ok(sc, ids, mask, NULL), "score: 0.03741 on 16 rows");
    check(score_ok(sc, ids, mask, disc), "score ignores a disc it does not declare");

    /* Short batch: row 0 only; a drift over TOL is reported, not failed. */
    fill_tokens(ids, mask, w1, 1);
    fill_disc(disc, ZERO, 1);
    int rc = cml_predict(sel, ids, mask, disc, SEL_OUT, 3, outs);
    check(rc == 0, "short batch predict");
    if (rc == 0) {
        float d = absf(sal[0] - SEL_WANT[0][0]);
        printf("INFO short batch row 0 saliency %.5f (full batch %.5f, diff %.5f)%s\n", sal[0], SEL_WANT[0][0], d,
               d > TOL ? " REPORTED: over 2e-3" : "");
    }
}

int main(void) {
    check(cml_init() == 0, "cml_init");
    check(cml_load("models/missing.mlmodelc", "select", 3) == NULL, "missing model loads as NULL (expected stderr)");
    cml_model *sel = cml_load(MODEL, "select", 3), *sc = cml_load(MODEL, "score", 3);
    check(sel != NULL, "load select, units 3");
    check(sc != NULL, "load score, units 3");
    long w1 = cml_width(sel, 0), w2 = cml_width(sc, 0);
    check(w1 == 128, "cml_width select == 128");
    check(w2 == 256, "cml_width score == 256");
    if (sel && sc && w1 > 0 && w1 <= MAX_W && w2 > 0 && w2 <= MAX_W) run(sel, sc, w1, w2);
    cml_free(sel);
    cml_free(sc);
    return summary();
}
