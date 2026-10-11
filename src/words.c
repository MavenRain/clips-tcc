/* whisper.cpp tokens -> TimedWords (SPEC 7.1) and the sentence splitter (SPEC 7.2). */
#include "words.h"
#include "uni.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ggml-backend.h>
#include <whisper.h>

enum { RUN_ON_LIMIT = 320 };

/* Growable byte string; always NUL-terminated once non-empty. */
typedef struct { char *p; size_t n, cap; } sbuf;

static int sbuf_add(sbuf *b, const char *s, size_t len)
{
    size_t need = b->n + len + 1, cap = b->cap ? b->cap : 64;
    if (need > b->cap) {
        while (cap < need) cap *= 2;
        char *p = realloc(b->p, cap);
        if (!p) return -1;
        b->p = p;
        b->cap = cap;
    }
    memcpy(b->p + b->n, s, len);
    b->n += len;
    b->p[b->n] = 0;
    return 0;
}

static char *dup_range(const char *s, size_t len)
{
    char *d = malloc(len + 1);
    if (!d) return NULL;
    memcpy(d, s, len);
    d[len] = 0;
    return d;
}

/* Grow *arr (elements of size sz) to hold at least need, doubling. */
static int grow(void **arr, size_t *cap, size_t need, size_t sz)
{
    size_t c = *cap ? *cap : 16;
    if (need <= *cap) return 0;
    while (c < need) c *= 2;
    void *p = realloc(*arr, c * sz);
    if (!p) return -1;
    *arr = p;
    *cap = c;
    return 0;
}

void words_free(cw_word *w, size_t n)
{
    if (!w) return;
    for (size_t i = 0; i < n; i++) free(w[i].text);
    free(w);
}

void sentences_free(cw_sentence *s, size_t n)
{
    if (!s) return;
    for (size_t i = 0; i < n; i++) free(s[i].text);
    free(s);
}

/* ---- whisper.cpp ---- */

static void quiet_log(enum ggml_log_level level, const char *text, void *user)
{
    (void)level;
    (void)text;
    (void)user;
}

/* Word list under construction; cur is the open word's text. */
typedef struct {
    cw_word *w;
    size_t n, cap;
    sbuf cur;
    double start, end;
    bool open;
} wlist;

/* Close the open word (TimedWord.init: end = max(start, end)). */
static int wl_close(wlist *l)
{
    if (!l->open) return 0;
    l->open = false;
    char *t = dup_range(l->cur.p ? l->cur.p : "", l->cur.n);
    l->cur.n = 0;
    if (!t || grow((void **)&l->w, &l->cap, l->n + 1, sizeof *l->w) < 0) {
        free(t);
        return -1;
    }
    l->w[l->n++] = (cw_word){ t, l->start, l->end > l->start ? l->end : l->start };
    return 0;
}

/* A token starting with a space opens a new word; any other token extends the open one. */
static int wl_token(wlist *l, const char *text, double t0, double t1)
{
    if (!l->open || text[0] == ' ') {
        if (wl_close(l) < 0) return -1;
        l->open = true;
        l->start = t0;
    }
    l->end = t1;
    return sbuf_add(&l->cur, text, strlen(text));
}

static int collect(struct whisper_context *ctx, cw_word **words, size_t *count)
{
    wlist l = {0};
    whisper_token eot = whisper_token_eot(ctx);
    int ns = whisper_full_n_segments(ctx), rc = 0;
    for (int s = 0; s < ns && rc == 0; s++) {
        int nt = whisper_full_n_tokens(ctx, s);
        for (int t = 0; t < nt && rc == 0; t++) {
            if (whisper_full_get_token_id(ctx, s, t) >= eot) continue;
            const char *text = whisper_full_get_token_text(ctx, s, t);
            whisper_token_data d = whisper_full_get_token_data(ctx, s, t);
            rc = text ? wl_token(&l, text, d.t0 / 100.0, d.t1 / 100.0) : -1;
        }
    }
    if (rc == 0) rc = wl_close(&l);
    free(l.cur.p);
    if (rc < 0) {
        words_free(l.w, l.n);
        fprintf(stderr, "words: out of memory\n");
        return -1;
    }
    *words = l.w;
    *count = l.n;
    return 0;
}

int words_transcribe(const char *model_path, const float *pcm, size_t n, cw_word **words, size_t *count)
{
    *words = NULL;
    *count = 0;
    if (n > INT_MAX) {
        fprintf(stderr, "words: %zu samples is too long\n", n);
        return -1;
    }
    whisper_log_set(quiet_log, NULL);
    /* Homebrew ggml loads its backends (CPU, Metal) dynamically; without them init aborts. */
    if (ggml_backend_dev_count() == 0) ggml_backend_load_all();
    /* CPU only: a Metal buffer allocation that fails (sandbox, no GPU) aborts inside ggml instead of returning NULL.
     * Unsandboxed Metal (whisper 1.9.5, ggml 0.26) aborts too, in whisper_init_state: "pre-allocated tensor
     * (leaf_0) in a buffer (MTL0) that cannot run the operation". */
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    struct whisper_context *ctx = whisper_init_from_file_with_params(model_path, cp);
    if (!ctx) {
        fprintf(stderr, "words: cannot load %s\n", model_path);
        return -1;
    }
    struct whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.language = "en";
    p.token_timestamps = true;
    p.print_special = false;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.n_threads = 4;
    int rc = whisper_full(ctx, p, pcm, (int)n) == 0 ? collect(ctx, words, count) : -1;
    if (rc < 0 && !*words) fprintf(stderr, "words: transcription failed\n");
    whisper_free(ctx);
    return rc;
}

/* ---- SPEC 7.2 splitter ---- */

static bool timed(const cw_word *w)
{
    return isfinite(w->start) && isfinite(w->end) && w->start >= 0;
}

/* Last grapheme cluster is exactly one of . ! ? U+2026. */
static bool terminated(const char *s, size_t len)
{
    size_t k = uni_last_grapheme(s, len);
    const char *g = s + k;
    size_t gl = len - k;
    return (gl == 1 && (g[0] == '.' || g[0] == '!' || g[0] == '?'))
        || (gl == 3 && memcmp(g, "\xE2\x80\xA6", 3) == 0);
}

typedef struct {
    sbuf text;
    double start, end;
    bool has_start;
} pending;

typedef struct {
    cw_sentence *s;
    size_t n, cap;
} slist;

/* Emit the trimmed pending text when nonempty and timed; always reset. */
static int take(pending *p, slist *out)
{
    size_t b = 0, e = 0;
    int rc = 0;
    if (p->text.n) uni_trim(p->text.p, p->text.n, &b, &e);
    if (e > b && p->has_start) {
        char *t = dup_range(p->text.p + b, e - b);
        if (!t || grow((void **)&out->s, &out->cap, out->n + 1, sizeof *out->s) < 0) {
            free(t);
            rc = -1;
        } else {
            out->s[out->n] = (cw_sentence){ t, p->start, p->end > p->start ? p->end : p->start };
            out->n++;
        }
    }
    p->text.n = 0;
    p->has_start = false;
    p->end = 0.0;
    return rc;
}

/* closesSentence: trimmed nonempty and (terminator last ? count > 1 : count >= limit). */
static bool closes(const pending *p)
{
    size_t b = 0, e = 0;
    if (!p->text.n) return false;
    uni_trim(p->text.p, p->text.n, &b, &e);
    if (e <= b) return false;
    long g = uni_graphemes(p->text.p + b, e - b);
    return terminated(p->text.p + b, e - b) ? g > 1 : g >= RUN_ON_LIMIT;
}

/* Every Character of s is whitespace (true for ""). */
static bool all_space(const char *s)
{
    size_t len = strlen(s), b = 0, e = 0;
    uni_trim(s, len, &b, &e);
    return e <= b;
}

int words_sentences(const cw_word *w, size_t n, cw_sentence **out, size_t *count)
{
    pending p = {0};
    slist l = {0};
    int rc = 0;
    *out = NULL;
    *count = 0;
    for (size_t i = 0; i < n && rc == 0; i++) {
        const char *t = w[i].text ? w[i].text : "";
        if (!p.has_start && all_space(t)) continue;
        rc = sbuf_add(&p.text, t, strlen(t));
        if (rc == 0 && timed(&w[i])) {
            if (!p.has_start) p.start = w[i].start;
            p.has_start = true;
            p.end = w[i].end;
        }
        if (rc == 0 && closes(&p)) rc = take(&p, &l);
    }
    if (rc == 0) rc = take(&p, &l);
    free(p.text.p);
    if (rc < 0) {
        sentences_free(l.s, l.n);
        fprintf(stderr, "words: out of memory\n");
        return -1;
    }
    *out = l.s;
    *count = l.n;
    return 0;
}
