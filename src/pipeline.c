/* Selector, candidates, scorer and selection (SPEC sections 2-5).
 * Mirrors desert-ant-core Sources/Clips/Pipeline.swift (P) and Model.swift (M). */
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utf8proc.h>

#include "features.h"
#include "pipeline.h"

enum { SENTENCE_TOKENS = 64, MIN_SENTENCES = 3, MAX_SENTENCES = 10, LOOKBACK = 8, LOOKAHEAD = 14,
       FALLBACK_WIDTH = 128 };

static const double WORDS_PER_SECOND = 2.5, JACCARD_MAX = 0.5, EPS = 1e-6;

/* Duration curve, P:35-41: minutes below the bound -> budget; else 14. */
static const struct { double below; int budget; } curve[] = {{5, 8}, {10, 11}, {30, 13}};

typedef struct { cp_span s; int idx; } keyed;

static int fail(const char *what)
{
    fprintf(stderr, "pipeline: %s\n", what);
    return -1;
}

int pipeline_budget(const char *const *sentences, int n, int limit)
{
    long words = 0;
    double minutes;
    if (limit > 0)
        return limit;
    for (int i = 0; i < n; i++) {
        long w = features_word_count(sentences[i]);
        words += w > 1 ? w : 1;
    }
    minutes = (double)words / WORDS_PER_SECOND / 60.0;
    for (size_t i = 0; i < sizeof curve / sizeof curve[0]; i++)
        if (minutes < curve[i].below)
            return curve[i].budget;
    return 14;
}

/* Saliency descending, lower index first on ties (Swift sorted is stable). */
static int by_saliency(const void *a, const void *b)
{
    const keyed *x = a, *y = b;
    if (x->s.score != y->s.score)
        return x->s.score > y->s.score ? -1 : 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

static int by_lo_hi(const void *a, const void *b)
{
    const cp_span *x = a, *y = b;
    if (x->lo != y->lo)
        return x->lo < y->lo ? -1 : 1;
    return (x->hi > y->hi) - (x->hi < y->hi);
}

int pipeline_candidates(const double *saliency, int n, int budget, cp_span **out, int *count)
{
    long na = (long)budget * 4;
    keyed *rank;
    cp_span *spans;
    int c = 0, k = 0;
    *out = NULL;
    *count = 0;
    if (na < 12)
        na = 12;
    if (na > n)
        na = n > 0 ? n : 0;
    rank = malloc((size_t)(n > 0 ? n : 1) * sizeof *rank);
    spans = malloc((size_t)(na > 0 ? na : 1) * (LOOKBACK + 1) * (LOOKAHEAD + 1) * sizeof *spans);
    if (!rank || !spans) {
        free(rank);
        free(spans);
        return fail("out of memory");
    }
    for (int i = 0; i < n; i++) {
        rank[i].s.lo = rank[i].s.hi = i;
        rank[i].s.score = saliency[i];
        rank[i].idx = i;
    }
    qsort(rank, (size_t)n, sizeof *rank, by_saliency);
    for (long r = 0; r < na; r++) {
        int a = rank[r].idx;
        for (int lo = a - LOOKBACK > 0 ? a - LOOKBACK : 0; lo <= a; lo++)
            for (int hi = a; hi <= a + LOOKAHEAD && hi < n; hi++) {
                int len = hi - lo + 1;
                if (len < MIN_SENTENCES || len > MAX_SENTENCES)
                    continue;
                spans[c].lo = lo;
                spans[c].hi = hi;
                spans[c].score = 0;
                c++;
            }
    }
    free(rank);
    qsort(spans, (size_t)c, sizeof *spans, by_lo_hi);
    for (int i = 0; i < c; i++)
        if (k == 0 || by_lo_hi(&spans[k - 1], &spans[i]) != 0)
            spans[k++] = spans[i];
    *out = spans;
    *count = k;
    return 0;
}

static int by_end(const void *a, const void *b)
{
    const keyed *x = a, *y = b;
    if (x->s.hi != y->s.hi)
        return x->s.hi < y->s.hi ? -1 : 1;
    if (x->s.lo != y->s.lo)
        return x->s.lo < y->s.lo ? -1 : 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

/* SPEC 5.1 (P:117-159). k above n changes nothing (dp[i][k] = dp[i][i] for k >= i), so it is capped. */
static int schedule(const cp_span *spans, int n, int budget, cp_span **out, int *nout)
{
    int kmax = budget < n ? budget : n, c = 0, bk = 0, i, k;
    size_t w = (size_t)kmax + 1, cells = ((size_t)n + 1) * w;
    keyed *s = malloc((size_t)n * sizeof *s);
    int *prev = malloc((size_t)n * sizeof *prev);
    double *dp = malloc(cells * sizeof *dp);
    unsigned char *take = calloc(cells, 1);
    cp_span *res = malloc((size_t)(kmax > 0 ? kmax : 1) * sizeof *res);
    const double neg = -DBL_MAX;
    double best;
    if (!s || !prev || !dp || !take || !res) {
        free(s);
        free(prev);
        free(dp);
        free(take);
        free(res);
        return fail("out of memory");
    }
    for (i = 0; i < n; i++) {
        s[i].s = spans[i];
        s[i].idx = i;
    }
    qsort(s, (size_t)n, sizeof *s, by_end);
    /* prev[i] = largest j < i with hi_j < lo_i; hi is ascending, so binary search. */
    for (i = 0; i < n; i++) {
        int lo = 0, hi = i;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (s[mid].s.hi < s[i].s.lo)
                lo = mid + 1;
            else
                hi = mid;
        }
        prev[i] = lo - 1;
    }
    for (k = 0; k <= kmax; k++)
        dp[k] = 0;
    for (i = 1; i <= n; i++) {
        size_t p = (size_t)(prev[i - 1] + 1);
        for (k = 0; k <= kmax; k++) {
            double skip = dp[(size_t)(i - 1) * w + (size_t)k];
            double base = k >= 1 ? dp[p * w + (size_t)k - 1] : neg;
            double use = k >= 1 && base > neg / 2 ? base + s[i - 1].s.score : neg;
            size_t at = (size_t)i * w + (size_t)k;
            take[at] = use > skip + EPS;
            dp[at] = take[at] ? use : skip;
        }
    }
    best = dp[(size_t)n * w];
    for (k = 1; k <= kmax; k++)
        if (dp[(size_t)n * w + (size_t)k] > best)
            best = dp[(size_t)n * w + (size_t)k];
    while (bk < kmax && dp[(size_t)n * w + (size_t)bk] < best - EPS)
        bk++;
    for (i = n, k = bk; i > 0 && k > 0;) {
        if (take[(size_t)i * w + (size_t)k]) {
            res[c++] = s[i - 1].s;
            k--;
            i = prev[i - 1] + 1;
        } else {
            i--;
        }
    }
    for (i = 0; i < c / 2; i++) {
        cp_span t = res[i];
        res[i] = res[c - 1 - i];
        res[c - 1 - i] = t;
    }
    free(s);
    free(prev);
    free(dp);
    free(take);
    *out = res;
    *nout = c;
    return 0;
}

/* Sentences lo..hi joined with one U+0020 (M:80). */
static char *join_span(const char *const *s, int lo, int hi)
{
    size_t len = 0, at = 0;
    char *out;
    for (int i = lo; i <= hi; i++)
        len += strlen(s[i]) + 1;
    out = malloc(len + 1);
    if (!out)
        return NULL;
    for (int i = lo; i <= hi; i++) {
        size_t k = strlen(s[i]);
        if (i > lo)
            out[at++] = ' ';
        memcpy(out + at, s[i], k);
        at += k;
    }
    out[at] = '\0';
    return out;
}

/* Next scalar of s[*i..len); an invalid byte reads as U+FFFD. */
static int32_t next_cp(const char *s, size_t len, size_t *i)
{
    utf8proc_int32_t cp = 0;
    utf8proc_ssize_t k = utf8proc_iterate((const utf8proc_uint8_t *)s + *i, (utf8proc_ssize_t)(len - *i), &cp);
    if (k <= 0) {
        *i += 1;
        return 0xFFFD;
    }
    *i += (size_t)k;
    return cp;
}

/* Same separator rule as features_word_count: a U+0020 that is a whole grapheme cluster. */
static int is_sep(const char *s, size_t len, size_t i)
{
    size_t k;
    if (s[i] != ' ')
        return 0;
    if (i > 0) {
        size_t j = i - 1;
        int32_t prev;
        while (j > 0 && i - j < 4 && ((unsigned char)s[j] & 0xC0) == 0x80)
            j--;
        k = j;
        prev = next_cp(s, len, &k);
        if (k != i)
            prev = 0xFFFD;
        if (!utf8proc_grapheme_break_stateful(prev, ' ', NULL))
            return 0;
    }
    if (i + 1 >= len)
        return 1;
    k = i + 1;
    return utf8proc_grapheme_break_stateful(' ', next_cp(s, len, &k), NULL);
}

/* Sorted, distinct words. Swift Set<String> equality is canonical equivalence, so words are NFC. */
typedef struct { char **w; long n; } wordset;

static void wordset_free(wordset *ws)
{
    for (long i = 0; i < ws->n; i++)
        free(ws->w[i]);
    free(ws->w);
    ws->w = NULL;
    ws->n = 0;
}

static char *nfc_copy(const char *b, size_t n)
{
    utf8proc_uint8_t *dst = NULL;
    char *raw;
    if (utf8proc_map((const utf8proc_uint8_t *)b, (utf8proc_ssize_t)n, &dst, UTF8PROC_STABLE | UTF8PROC_COMPOSE) >= 0)
        return (char *)dst;
    raw = malloc(n + 1);
    if (raw) {
        memcpy(raw, b, n);
        raw[n] = '\0';
    }
    return raw;
}

static int by_bytes(const void *a, const void *b)
{
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static int wordset_build(const char *const *sentences, int lo, int hi, wordset *ws)
{
    long cap = 0, k = 0;
    ws->w = NULL;
    ws->n = 0;
    for (int at = lo; at <= hi; at++)
        cap += features_word_count(sentences[at]);
    ws->w = malloc((size_t)(cap > 0 ? cap : 1) * sizeof *ws->w);
    if (!ws->w)
        return -1;
    for (int at = lo; at <= hi; at++) {
        const char *text = sentences[at];
        size_t len = strlen(text), i = 0;
        while (i < len) {
            size_t b;
            char *w;
            while (i < len && is_sep(text, len, i))
                i++;
            if (i >= len)
                break;
            b = i;
            while (i < len && !is_sep(text, len, i))
                i++;
            w = nfc_copy(text + b, i - b);
            if (!w) {
                wordset_free(ws);
                return -1;
            }
            ws->w[ws->n++] = w;
        }
    }
    qsort(ws->w, (size_t)ws->n, sizeof *ws->w, by_bytes);
    for (long j = 0; j < ws->n; j++) {
        if (k > 0 && strcmp(ws->w[k - 1], ws->w[j]) == 0)
            free(ws->w[j]);
        else
            ws->w[k++] = ws->w[j];
    }
    ws->n = k;
    return 0;
}

/* SPEC 5.2: |A n B| / max(|A u B|, 1) > 0.5. */
static int too_similar(const wordset *a, const wordset *b)
{
    long i = 0, j = 0, inter = 0, uni;
    while (i < a->n && j < b->n) {
        int c = strcmp(a->w[i], b->w[j]);
        inter += c == 0;
        i += c <= 0;
        j += c >= 0;
    }
    uni = a->n + b->n - inter;
    return (double)inter / (double)(uni > 0 ? uni : 1) > JACCARD_MAX;
}

/* SPEC 5.2 (P:161-174) over spans in time order; an earlier span wins. */
static int near_cut(const cp_span *ch, int n, const char *const *sentences, cp_span **out, int *kept)
{
    wordset *sets = calloc((size_t)(n > 0 ? n : 1), sizeof *sets);
    cp_span *res = malloc((size_t)(n > 0 ? n : 1) * sizeof *res);
    int nk = 0, rc = sets && res ? 0 : -1;
    for (int i = 0; rc == 0 && i < n; i++) {
        int drop = 0;
        if (wordset_build(sentences, ch[i].lo, ch[i].hi, &sets[nk]) != 0) {
            rc = -1;
            break;
        }
        for (int j = 0; j < nk && !drop; j++)
            drop = too_similar(&sets[nk], &sets[j]);
        if (drop) {
            wordset_free(&sets[nk]);
            continue;
        }
        res[nk++] = ch[i];
    }
    for (int j = 0; sets && j < nk; j++)
        wordset_free(&sets[j]);
    free(sets);
    if (rc != 0) {
        free(res);
        return fail("out of memory");
    }
    *out = res;
    *kept = nk;
    return 0;
}

int pipeline_select(const cp_span *spans, int count, int budget, const char *const *sentences,
                    cp_span **out, int *kept)
{
    cp_span *chosen = NULL;
    int nch = 0, rc;
    *out = NULL;
    *kept = 0;
    if (count <= 0 || budget <= 0) {
        *out = malloc(sizeof **out);
        return *out ? 0 : fail("out of memory");
    }
    if (schedule(spans, count, budget, &chosen, &nch) != 0)
        return -1;
    rc = near_cut(chosen, nch, sentences, out, kept);
    free(chosen);
    return rc;
}

/* SPEC 2 (M:86-115): one select call per 16 sentences; rows past a short batch stay zero. */
static int run_selector(cml_model *m, const tok *t, const char *const *sentences, int n, double *sal)
{
    long w = cml_width(m, FALLBACK_WIDTH);
    int max_len = w < SENTENCE_TOKENS ? (int)w : SENTENCE_TOKENS, rc = 0;
    size_t cells = (size_t)CML_BATCH * (size_t)(w > 0 ? w : 1);
    int32_t *ids = malloc(cells * sizeof *ids), *mask = malloc(cells * sizeof *mask);
    float disc[CML_BATCH * CML_DISC], o_sal[CML_BATCH], o_start[CML_BATCH], o_end[CML_BATCH];
    const char *const names[3] = {"saliency", "start_p", "end_p"};
    float *const outs[3] = {o_sal, o_start, o_end};
    if (w < 1 || !ids || !mask)
        rc = fail(w < 1 ? "selector width unknown" : "out of memory");
    for (int start = 0; rc == 0 && start < n; start += CML_BATCH) {
        int rows = n - start < CML_BATCH ? n - start : CML_BATCH;
        memset(ids, 0, cells * sizeof *ids);
        memset(mask, 0, cells * sizeof *mask);
        memset(disc, 0, sizeof disc);
        for (int r = 0; rc == 0 && r < rows; r++) {
            int k = tok_encode(t, sentences[start + r], max_len, ids + (size_t)r * (size_t)w);
            if (k < 0 || features_disc(sentences[start + r], start + r, n, disc + r * CML_DISC) != 0) {
                rc = fail("selector row failed");
                break;
            }
            for (int c = 0; c < k; c++)
                mask[(size_t)r * (size_t)w + (size_t)c] = 1;
        }
        if (rc == 0 && cml_predict(m, ids, mask, disc, names, 3, outs) != 0)
            rc = fail("select predict failed");
        for (int r = 0; rc == 0 && r < rows; r++)
            sal[start + r] = (double)o_sal[r];
    }
    free(ids);
    free(mask);
    return rc;
}

/* SPEC 4 (M:121-141): span texts at the full scorer width, output "score", raw. */
static int run_scorer(cml_model *m, const tok *t, const char *const *sentences, cp_span *cand, int nc)
{
    long w = cml_width(m, FALLBACK_WIDTH);
    size_t cells = (size_t)CML_BATCH * (size_t)(w > 0 ? w : 1);
    int32_t *ids = malloc(cells * sizeof *ids), *mask = malloc(cells * sizeof *mask);
    float o_score[CML_BATCH];
    const char *const names[1] = {"score"};
    float *const outs[1] = {o_score};
    int rc = 0;
    if (w < 1 || w > 0x7fffffffL || !ids || !mask)
        rc = fail(w < 1 || w > 0x7fffffffL ? "scorer width unknown" : "out of memory");
    for (int start = 0; rc == 0 && start < nc; start += CML_BATCH) {
        int rows = nc - start < CML_BATCH ? nc - start : CML_BATCH;
        memset(ids, 0, cells * sizeof *ids);
        memset(mask, 0, cells * sizeof *mask);
        for (int r = 0; rc == 0 && r < rows; r++) {
            char *text = join_span(sentences, cand[start + r].lo, cand[start + r].hi);
            int k = text ? tok_encode(t, text, (int)w, ids + (size_t)r * (size_t)w) : -1;
            free(text);
            if (k < 0) {
                rc = fail("scorer row failed");
                break;
            }
            for (int c = 0; c < k; c++)
                mask[(size_t)r * (size_t)w + (size_t)c] = 1;
        }
        if (rc == 0 && cml_predict(m, ids, mask, NULL, names, 1, outs) != 0)
            rc = fail("score predict failed");
        for (int r = 0; rc == 0 && r < rows; r++)
            cand[start + r].score = (double)o_score[r];
    }
    free(ids);
    free(mask);
    return rc;
}

static int by_score(const void *a, const void *b)
{
    const keyed *x = a, *y = b;
    if (x->s.score != y->s.score)
        return x->s.score > y->s.score ? -1 : 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

/* SPEC 5.3 (P:85-96): stable score sort, then the Clip fields. */
static int make_clips(const cp_span *kept, int nk, const char *const *sentences, cp_clip **clips, int *count)
{
    keyed *k = malloc((size_t)(nk > 0 ? nk : 1) * sizeof *k);
    cp_clip *c = calloc((size_t)(nk > 0 ? nk : 1), sizeof *c);
    if (!k || !c) {
        free(k);
        free(c);
        return fail("out of memory");
    }
    for (int i = 0; i < nk; i++) {
        k[i].s = kept[i];
        k[i].idx = i;
    }
    qsort(k, (size_t)nk, sizeof *k, by_score);
    for (int i = 0; i < nk; i++) {
        c[i].id = i;
        c[i].lo = k[i].s.lo;
        c[i].hi = k[i].s.hi;
        c[i].score = k[i].s.score;
        c[i].percentile = nk > 1 ? 1.0 - (double)i / (double)(nk - 1) : 1.0;
        c[i].text = join_span(sentences, c[i].lo, c[i].hi);
        if (!c[i].text) {
            free(k);
            pipeline_free(c, nk);
            return fail("out of memory");
        }
        long words = 0;
        for (int j = c[i].lo; j <= c[i].hi; j++)
            words += features_word_count(sentences[j]);
        c[i].est_duration_sec = (double)words / WORDS_PER_SECOND;
    }
    free(k);
    *clips = c;
    *count = nk;
    return 0;
}

int pipeline_run(cml_model *select, cml_model *score, const tok *t, const char *const *sentences, int n,
                 int limit, cp_clip **clips, int *count)
{
    double *sal;
    cp_span *cand = NULL, *kept = NULL;
    int budget, nc = 0, nk = 0, rc;
    *clips = NULL;
    *count = 0;
    if (n < MIN_SENTENCES)
        return 0;
    budget = pipeline_budget(sentences, n, limit);
    sal = malloc((size_t)n * sizeof *sal);
    if (!sal)
        return fail("out of memory");
    rc = run_selector(select, t, sentences, n, sal);
    if (rc == 0)
        rc = pipeline_candidates(sal, n, budget, &cand, &nc);
    free(sal);
    if (rc == 0 && nc > 0)
        rc = run_scorer(score, t, sentences, cand, nc);
    if (rc == 0 && nc > 0)
        rc = pipeline_select(cand, nc, budget, sentences, &kept, &nk);
    free(cand);
    if (rc == 0 && nk > 0)
        rc = make_clips(kept, nk, sentences, clips, count);
    free(kept);
    return rc;
}

void pipeline_free(cp_clip *c, int count)
{
    if (!c)
        return;
    for (int i = 0; i < count; i++)
        free(c[i].text);
    free(c);
}
