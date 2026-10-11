/* Stage 4 gate: discourse features, budget, candidates, selection, and a stubbed pipeline_run. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coreml.h"
#include "features.h"
#include "pipeline.h"
#include "tokenizer.h"

static int checks, failed;

static void check(int ok, const char *what)
{
    checks++;
    if (!ok) {
        failed++;
        fprintf(stderr, "FAIL: %s\n", what);
    }
}

/* Stubs for the Core ML and tokenizer symbols pipeline.o references. */
struct cml_model { int scorer; long calls, pad_rows, bad; };

long cml_width(const cml_model *m, long fallback)
{
    (void)fallback;
    return m->scorer ? 256 : 128;
}

int tok_encode(const tok *t, const char *utf8, int max_len, int32_t *out)
{
    int n = (int)(strlen(utf8) / 4) + 2;
    (void)t;
    if (max_len <= 0)
        return 0;
    if (n > max_len)
        n = max_len;
    for (int i = 0; i < n; i++)
        out[i] = i == 0 ? 0 : (i == n - 1 ? 2 : 100 + i);
    return n;
}

int cml_predict(cml_model *m, const int32_t *ids, const int32_t *mask, const float *disc,
                const char *const *names, int n_out, float *const *outs)
{
    long w = cml_width(m, 128);
    int wrong = m->scorer ? (disc != NULL || n_out != 1 || strcmp(names[0], "score") != 0)
                          : (disc == NULL || n_out != 3 || strcmp(names[0], "saliency") != 0
                             || strcmp(names[1], "start_p") != 0 || strcmp(names[2], "end_p") != 0);
    m->calls++;
    m->bad += wrong;
    for (int r = 0; r < CML_BATCH; r++) {
        long toks = 0, nz = 0;
        for (long c = 0; c < w; c++) {
            toks += mask[r * w + c];
            nz += ids[r * w + c] != 0;
        }
        for (int c = 0; disc && c < CML_DISC; c++)
            nz += disc[r * CML_DISC + c] != 0.0f;
        if (toks == 0) {
            m->pad_rows++;
            m->bad += nz != 0;
        }
        for (int k = 0; k < n_out; k++)
            outs[k][r] = m->scorer ? 0.1f * (float)toks : (disc ? 1.0f - disc[r * CML_DISC] : 0.0f);
    }
    return 0;
}

static void cues(const char *s, float hook, float payoff, float endq, const char *what)
{
    float f[5] = {0};
    int rc = features_disc(s, 0, 1, f), ok = rc == 0 && f[1] == hook && f[2] == payoff && f[3] == endq;
    check(ok, what);
    if (!ok)
        fprintf(stderr, "  rc %d hook %g payoff %g q %g\n", rc, f[1], f[2], f[3]);
}

static void test_features(void)
{
    float f[5] = {0};
    cues("Here's the secret: nobody knows why.", 5, 0, 0, "hook sentence: here's, nobody x2, the secret, why");
    cues("So that's why the result matters, which means everything.", 1, 4, 0, "payoff sentence");
    cues("whyever", 0, 0, 0, "whyever has no \\bwhy\\b");
    cues("why?", 2, 0, 1, "why? matches \\bwhy\\b and \\?");
    cues("the mistakes", 1, 0, 0, "the mistake has no \\b");
    cues("Why?  \t \xe3\x80\x80", 2, 0, 1, "ends_with_question after trailing White_Space");
    cues("? .", 1, 0, 0, "? not last after trim");
    cues("here\xe2\x80\x99s the deal", 0, 0, 0, "U+2019 is not folded to an apostrophe");
    cues("heres the deal", 1, 0, 0, "here'?s without the apostrophe");
    cues("thats why", 1, 1, 0, "that'?s why without the apostrophe");
    cues("SO THAT'S WHY", 1, 2, 0, "upper case is lowercased first");
    cues("HOW \xc4\xb0", 0, 0, 0, "dotted I retains its combining dot in cue matching");
    cues("HOW I", 1, 0, 0, "ASCII I retains the how I cue");
    cues("so\xcc\x81 good", 0, 0, 0, "no \\b before a combining mark (ICU)");
    cues("x_so 1so", 0, 0, 0, "_ and digits are word characters");
    cues("so-called", 0, 1, 0, "- is not a word character");
    cues("\xc5\xbfo it goes", 0, 1, 0, "long s folds to s (case-insensitive)");
    cues("most people", 2, 0, 0, "most people and \\bmost\\b");
    check(features_disc("abc 12", 0, 1, f) == 0 && f[4] == 2.0f / 7.0f, "digit_ratio(abc 12) = 2/7");
    check(features_disc("\xc2\xbd \xd9\xa3 \xe2\x85\xa3", 0, 1, f) == 0 && f[4] == 3.0f / 6.0f,
          "digit_ratio counts No, Nd and Nl clusters");
    check(features_disc("x", 3, 4, f) == 0 && f[0] == 0.75f, "relative_position 3/4");
    check(features_disc("x", 0, 0, f) == 0 && f[0] == 0.0f, "relative_position with n = 0");
    check(features_word_count("  a  b c ") == 3, "word count skips empty tokens");
    check(features_word_count("") == 0, "word count of empty text");
    check(features_word_count("a \xcc\x81" "b") == 1, "space + combining mark is no separator");
}

static void test_budget(void)
{
    static const int sizes[] = {749, 750, 1499, 1500, 4499, 4500}, want[] = {8, 11, 11, 13, 13, 14};
    const char **many = malloc(4500 * sizeof *many), **empty = malloc(750 * sizeof *empty);
    if (!many || !empty) {
        free(many);
        free(empty);
        check(0, "budget: out of memory");
        return;
    }
    for (int i = 0; i < 4500; i++)
        many[i] = "w";
    for (int i = 0; i < 750; i++)
        empty[i] = "";
    check(pipeline_budget((const char *const *)many, 3, 3) == 3, "budget: limit 3 -> 3");
    for (int i = 0; i < 6; i++) {
        char what[64];
        snprintf(what, sizeof what, "budget: %d words -> %d", sizes[i], want[i]);
        check(pipeline_budget((const char *const *)many, sizes[i], 0) == want[i], what);
    }
    check(pipeline_budget((const char *const *)empty, 750, 0) == 11, "budget: empty sentence counts 1 word");
    free(many);
    free(empty);
}

/* Independent oracle: stable anchors by repeated selection, then every 3..10 span that holds one. */
static int oracle(const double *sal, int n, int budget, int *los, int *his)
{
    int na = budget * 4 > 12 ? budget * 4 : 12, used[64] = {0}, anchor[64], c = 0;
    if (na > n)
        na = n;
    for (int k = 0; k < na; k++) {
        int best = -1;
        for (int i = 0; i < n; i++)
            if (!used[i] && (best < 0 || sal[i] > sal[best]))
                best = i;
        used[best] = 1;
        anchor[k] = best;
    }
    for (int lo = 0; lo < n; lo++)
        for (int hi = lo + 2; hi < n && hi - lo < 10; hi++) {
            int hit = 0;
            for (int k = 0; k < na && !hit; k++)
                hit = anchor[k] >= lo && anchor[k] <= hi && anchor[k] - lo <= 8 && hi - anchor[k] <= 14;
            if (hit) {
                los[c] = lo;
                his[c] = hi;
                c++;
            }
        }
    return c;
}

static void cand_case(const double *sal, int n, int budget, int want, const char *what)
{
    cp_span *c = NULL;
    int nc = -1, los[1024], his[1024];
    int rc = pipeline_candidates(sal, n, budget, &c, &nc), no = oracle(sal, n, budget, los, his);
    int ok = rc == 0 && nc == no && (want < 0 || nc == want);
    for (int i = 0; ok && i < nc; i++)
        ok = c[i].lo == los[i] && c[i].hi == his[i] && c[i].hi - c[i].lo + 1 >= 3 && c[i].hi - c[i].lo + 1 <= 10;
    for (int i = 1; ok && i < nc; i++)
        ok = c[i - 1].lo < c[i].lo || (c[i - 1].lo == c[i].lo && c[i - 1].hi < c[i].hi);
    check(ok, what);
    if (!ok)
        fprintf(stderr, "  rc %d count %d oracle %d want %d\n", rc, nc, no, want);
    free(c);
}

static void test_candidates(void)
{
    double desc[20], flat[20], mixed[20];
    for (int i = 0; i < 20; i++) {
        desc[i] = 20 - i;
        flat[i] = 0.5;
        mixed[i] = (double)((i * 7) % 5);
    }
    cand_case(desc, 20, 1, 95, "candidates: anchors 0..11 give 95 spans");
    cand_case(flat, 20, 1, 95, "candidates: ties keep the lower index (stable)");
    cand_case(mixed, 20, 1, -1, "candidates: tied saliency matches the oracle");
    cand_case(mixed, 20, 5, 116, "candidates: 20 anchors give every 3..10 span");
    cand_case(desc, 2, 1, 0, "candidates: 2 sentences give none");
}

static void sel_case(const cp_span *in, int n, int budget, const char *const *s, int want, const int *pairs,
                     const char *what)
{
    cp_span *out = NULL;
    int kept = -1, rc = pipeline_select(in, n, budget, s, &out, &kept), ok = rc == 0 && kept == want;
    for (int i = 0; ok && i < kept; i++)
        ok = out[i].lo == pairs[2 * i] && out[i].hi == pairs[2 * i + 1];
    check(ok, what);
    if (!ok)
        fprintf(stderr, "  rc %d kept %d want %d\n", rc, kept, want);
    free(out);
}

static void test_select(void)
{
    static const char *const uniq[] = {"a0 b0", "a1 b1", "a2 b2", "a3 b3", "a4 b4", "a5 b5",
                                       "a6 b6", "a7 b7", "a8 b8", "a9 b9", "a10 b10", "a11 b11"};
    static const char *const dup[] = {"alpha beta", "gamma delta", "eps zeta", "alpha  beta", "gamma delta", "zeta eps"};
    static const char *const half[] = {"a", "b", "c", "a", "b", "d"};
    const cp_span two_win[] = {{0, 9, 10.0}, {0, 4, 6.0}, {5, 9, 6.0}};
    const cp_span long_win[] = {{0, 9, 10.0}, {0, 4, 4.0}, {5, 9, 4.0}};
    const cp_span negative[] = {{0, 2, -1.0}, {3, 5, -2.0}, {6, 8, 0.0}};
    const cp_span equal[] = {{6, 8, 1.0}, {0, 2, 1.0}, {3, 5, 1.0}};
    const cp_span shared[] = {{4, 8, 6.0}, {0, 4, 5.0}};
    const cp_span near[] = {{3, 5, 3.0}, {0, 2, 2.0}};
    const cp_span edge[] = {{0, 2, 1.0}, {3, 5, 1.0}};
    const cp_span combining[] = {{0, 2, 2.0}, {3, 5, 1.0}};
    static const char *const combining_s[] = {"a", "\xcc\x81", "b", "b", "\xcc\x81", "a"};
    static const int p_two[] = {0, 4, 5, 9}, p_long[] = {0, 9}, p_cap[] = {0, 2, 3, 5}, p_shared[] = {4, 8},
                     p_near[] = {0, 2}, p_edge[] = {0, 2, 3, 5};
    sel_case(two_win, 3, 3, uniq, 2, p_two, "select: two disjoint 6s beat one 10");
    sel_case(long_win, 3, 3, uniq, 1, p_long, "select: one 10 beats two disjoint 4s");
    sel_case(negative, 3, 3, uniq, 0, NULL, "select: scores <= eps give none");
    sel_case(equal, 3, 2, uniq, 2, p_cap, "select: k cap 2 of 3, time order");
    sel_case(shared, 2, 2, uniq, 1, p_shared, "select: spans sharing sentence 4 never both");
    sel_case(two_win, 3, 0, uniq, 0, NULL, "select: budget 0 gives none");
    sel_case(near, 2, 2, dup, 1, p_near, "select: later near-duplicate dropped (Jaccard 1.0)");
    sel_case(edge, 2, 2, half, 2, p_edge, "select: Jaccard exactly 0.5 is kept");
    sel_case(combining, 2, 2, combining_s, 1, p_edge,
             "select: equal per-sentence word sets dedupe across combining marks");
}

static void test_run(void)
{
    char buf[20][48];
    const char *s[20];
    struct cml_model sel = {0, 0, 0, 0}, sco = {1, 0, 0, 0}, a = {0, 0, 0, 0}, b = {1, 0, 0, 0};
    cp_clip *c = NULL;
    int n = -1, rc, ok;
    for (int i = 0; i < 20; i++) {
        snprintf(buf[i], sizeof buf[i], "w%d alpha%d beta%d%s", i, i, i, i % 3 ? " gamma delta" : "");
        s[i] = buf[i];
    }
    rc = pipeline_run(&sel, &sco, NULL, s, 20, 3, &c, &n);
    check(rc == 0 && n >= 1 && n <= 3, "run: 1..3 clips at limit 3");
    check(sel.calls == 2 && sel.pad_rows == 12 && sel.bad == 0, "run: 2 select batches, 12 zero rows, disc given");
    check(sco.calls == 6 && sco.pad_rows == 1 && sco.bad == 0, "run: 95 candidates in 6 score batches, no disc");
    ok = rc == 0;
    for (int i = 0; ok && i < n; i++) {
        char want[1024];
        size_t at = 0;
        for (int j = c[i].lo; j <= c[i].hi; j++)
            at += (size_t)snprintf(want + at, sizeof want - at, j > c[i].lo ? " %s" : "%s", s[j]);
        ok = c[i].id == i && strcmp(c[i].text, want) == 0
          && c[i].est_duration_sec == (double)features_word_count(want) / 2.5
          && (i == 0 || c[i - 1].score >= c[i].score);
        for (int j = 0; ok && j < i; j++)
            ok = c[j].hi < c[i].lo || c[i].hi < c[j].lo;
    }
    ok = ok && (n < 1 || c[0].percentile == 1.0) && (n < 2 || c[n - 1].percentile == 0.0);
    check(ok, "run: ranks, text, duration, score order, no overlap, percentile");
    pipeline_free(c, n);
    c = NULL;
    rc = pipeline_run(&a, &b, NULL, s, 2, 0, &c, &n);
    check(rc == 0 && n == 0 && c == NULL && a.calls == 0 && b.calls == 0, "run: n < 3 gives 0 clips, no inference");
    {
        static const char *const combining_s[] = {"a", "\xcc\x81", "b"};
        struct cml_model sel_one = {0, 0, 0, 0}, sco_one = {1, 0, 0, 0};
        c = NULL;
        rc = pipeline_run(&sel_one, &sco_one, NULL, combining_s, 3, 1, &c, &n);
        check(rc == 0 && n == 1 && c[0].est_duration_sec == 3.0 / 2.5,
              "run: duration sums word counts before joining sentences");
        if (rc != 0 || n != 1 || c[0].est_duration_sec != 3.0 / 2.5)
            fprintf(stderr, "  rc %d count %d duration %g\n", rc, n, n > 0 ? c[0].est_duration_sec : 0.0);
        pipeline_free(c, n);
    }
}

int main(void)
{
    test_features();
    test_budget();
    test_candidates();
    test_select();
    test_run();
    printf("test_pipeline: %d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}
