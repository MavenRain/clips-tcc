/* clips: ranked, non-overlapping highlight spans of a video (SPEC 7.3-7.4, CLI ClipPipeline.swift:95-145).
 * Usage: clips VIDEO [--limit N] [--json] [--models DIR]. Run unsandboxed (Core ML cache).
 * Spans or the JSON report go to stdout; progress and timing go to stderr. */
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "audio.h"
#include "coreml.h"
#include "pipeline.h"
#include "tokenizer.h"
#include "words.h"

enum { UNITS_ALL = 2, PATH_LEN = 4096, LIMIT_MAX = 1000 };
#define PADDING 0.15 /* SPEC 7.3 default padding, seconds */

typedef struct { const char *video, *models; int limit, json; } args;
typedef struct { char model[PATH_LEN], vocab[PATH_LEN], whisper[PATH_LEN]; } paths;
typedef struct { double start, end; } range;

/* Pick (ClipPipeline.swift:110-113): first range start, last range end, summed range durations. */
typedef struct { double start, end, duration; } pick;

typedef struct {
    float *pcm;
    size_t n_pcm, n_words, n_sent;
    cw_word *words;
    cw_sentence *sent;
    const char **texts;
    cml_model *select, *score;
    tok *tok;
    cp_clip *clips;
    int n_clips;
    double material_sec, transcribe_sec, select_sec;
} state;

static int usage(void)
{
    fprintf(stderr, "usage: clips VIDEO [--limit N] [--json] [--models DIR]\n");
    return 2;
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static int parse_limit(const char *s, int *out)
{
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!*s || *end || v < 1 || v > LIMIT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int parse_args(int argc, char **argv, args *a)
{
    a->video = NULL;
    a->models = "models";
    a->limit = 0; /* 0: the duration curve picks the budget (SPEC 5) */
    a->json = 0;
    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
        if (!strcmp(s, "--json")) {
            a->json = 1;
            continue;
        }
        if (!strcmp(s, "--limit")) {
            if (i + 1 >= argc || parse_limit(argv[++i], &a->limit)) return -1;
            continue;
        }
        if (!strcmp(s, "--models")) {
            if (i + 1 >= argc) return -1;
            a->models = argv[++i];
            continue;
        }
        if (s[0] == '-' || a->video) return -1;
        a->video = s;
    }
    return a->video ? 0 : -1;
}

static int join(char *out, const char *dir, const char *file)
{
    int k = snprintf(out, PATH_LEN, "%s/%s", dir, file);
    return k < 0 || k >= PATH_LEN ? -1 : 0;
}

/* SPEC 7.3 share(gap): half the gap, at most PADDING, none for touching or overlapping sentences. */
static double half_gap(double gap)
{
    if (gap <= 0) return 0;
    return gap / 2 < PADDING ? gap / 2 : PADDING;
}

/* SPEC 7.3: the padded range of sentence id; no next sentence pads the end by PADDING. */
static range padded(const cw_sentence *s, size_t n, size_t id)
{
    double pre = half_gap(id > 0 ? s[id].start - s[id - 1].end : s[id].start);
    double post = id + 1 < n ? half_gap(s[id + 1].start - s[id].end) : PADDING;
    range r = { s[id].start - pre, s[id].end + post };
    if (r.start < 0) r.start = 0;
    if (r.end < r.start) r.end = r.start;
    return r;
}

/* SPEC 7.3 merge in id order: a range that starts at or before the open one's end extends it. */
static pick make_pick(const cp_clip *c, const cw_sentence *s, size_t n)
{
    pick p = { 0, 0, 0 };
    range cur = { 0, 0 };
    int open = 0;
    for (int id = c->lo; id <= c->hi; id++) {
        if (id < 0 || (size_t)id >= n) continue;
        range r = padded(s, n, (size_t)id);
        if (open && r.start <= cur.end) {
            if (r.end > cur.end) cur.end = r.end;
            continue;
        }
        if (open) p.duration += cur.end - cur.start;
        if (!open) p.start = r.start;
        cur = r;
        open = 1;
    }
    if (!open) return p;
    p.duration += cur.end - cur.start;
    p.end = cur.end;
    return p;
}

/* Shortest %g that reads back as v; non-finite values print as null. */
static void put_num(double v)
{
    char buf[40];
    if (v != v || v > DBL_MAX || v < -DBL_MAX) {
        fputs("null", stdout);
        return;
    }
    for (int prec = 15; prec <= 17; prec++) {
        snprintf(buf, sizeof buf, "%.*g", prec, v);
        if (strtod(buf, NULL) == v) break;
    }
    fputs(buf, stdout);
}

/* JSON string, slashes and non-ASCII kept as is (.withoutEscapingSlashes). */
static void put_str(const char *s)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char ch = *p;
        const char *esc = ch == '"' ? "\\\"" : ch == '\\' ? "\\\\" : ch == '\n' ? "\\n"
                        : ch == '\r' ? "\\r" : ch == '\t' ? "\\t" : NULL;
        if (esc) {
            fputs(esc, stdout);
            continue;
        }
        if (ch < 0x20) {
            printf("\\u%04x", ch);
            continue;
        }
        putchar(ch);
    }
    putchar('"');
}

static void print_text(const state *st)
{
    for (int i = 0; i < st->n_clips; i++) {
        const cp_clip *c = &st->clips[i];
        pick p = make_pick(c, st->sent, st->n_sent);
        printf("%2d  %8.2f  %8.2f  %.4f  %s\n", i + 1, p.start, p.end, c->score, c->text);
    }
    if (!st->n_clips) fprintf(stderr, "clips: no clips\n");
}

/* ClipReport (ClipPipeline.swift:20-29) as JSONEncoder [.prettyPrinted, .sortedKeys] writes it.
 * Additions: Pick.sentenceIDs (the clip's sentence ids). Omitted: transcript, Pick.file (no cut).
 * materialSec is the container duration, as the CLI reports it (talk.mp4: 94 s, audio track 92.15 s). */
static void print_json(const args *a, const state *st)
{
    printf("{\n  \"clips\" : [\n");
    if (!st->n_clips) printf("\n");
    for (int i = 0; i < st->n_clips; i++) {
        const cp_clip *c = &st->clips[i];
        pick p = make_pick(c, st->sent, st->n_sent);
        printf("    {\n      \"durationSec\" : ");
        put_num(p.duration);
        printf(",\n      \"end\" : ");
        put_num(p.end);
        printf(",\n      \"id\" : %d,\n      \"score\" : ", i + 1);
        put_num(c->score);
        printf(",\n      \"sentenceIDs\" : [\n");
        for (int id = c->lo; id <= c->hi; id++) printf("        %d%s\n", id, id < c->hi ? "," : "");
        printf("      ],\n      \"start\" : ");
        put_num(p.start);
        printf(",\n      \"text\" : ");
        put_str(c->text);
        printf("\n    }%s\n", i + 1 < st->n_clips ? "," : "");
    }
    printf("  ],\n  \"cutSec\" : 0,\n  \"input\" : ");
    put_str(a->video);
    printf(",\n  \"materialSec\" : ");
    put_num(st->material_sec);
    printf(",\n  \"selectSec\" : ");
    put_num(st->select_sec);
    printf(",\n  \"sentences\" : %zu,\n  \"transcribeSec\" : ", st->n_sent);
    put_num(st->transcribe_sec);
    printf("\n}\n");
}

/* audio -> words -> sentences -> models -> pipeline_run. Errors to stderr; st owns everything. */
static int run(const args *a, const paths *p, state *st)
{
    double t0 = now();
    if (audio_load(a->video, &st->pcm, &st->n_pcm, &st->material_sec)) {
        fprintf(stderr, "clips: cannot decode the audio of %s\n", a->video);
        return 1;
    }
    fprintf(stderr, "clips: audio %.2f s, media %.2f s\n", (double)st->n_pcm / AUDIO_RATE, st->material_sec);
    if (words_transcribe(p->whisper, st->pcm, st->n_pcm, &st->words, &st->n_words)) return 1;
    free(st->pcm);
    st->pcm = NULL;
    if (words_sentences(st->words, st->n_words, &st->sent, &st->n_sent)) return 1;
    st->transcribe_sec = now() - t0;
    fprintf(stderr, "clips: %zu words, %zu sentences, transcribe %.2f s\n", st->n_words, st->n_sent,
            st->transcribe_sec);
    if (st->n_sent < 3 || st->n_sent > INT_MAX) {
        fprintf(stderr, "clips: %zu sentences; the clips model needs at least three\n", st->n_sent);
        return 1;
    }

    double t1 = now();
    if (cml_init()) return 1;
    st->select = cml_load(p->model, "select", UNITS_ALL);
    if (!st->select) return 1;
    st->score = cml_load(p->model, "score", UNITS_ALL);
    if (!st->score) return 1;
    st->tok = tok_load(p->vocab);
    if (!st->tok) {
        fprintf(stderr, "clips: cannot load %s\n", p->vocab);
        return 1;
    }
    st->texts = malloc(st->n_sent * sizeof *st->texts);
    if (!st->texts) {
        fprintf(stderr, "clips: out of memory\n");
        return 1;
    }
    for (size_t i = 0; i < st->n_sent; i++) st->texts[i] = st->sent[i].text;
    if (pipeline_run(st->select, st->score, st->tok, st->texts, (int)st->n_sent, a->limit, &st->clips,
                     &st->n_clips)) {
        fprintf(stderr, "clips: selection failed\n");
        return 1;
    }
    st->select_sec = now() - t1;
    fprintf(stderr, "clips: %d clips, select %.2f s\n", st->n_clips, st->select_sec);
    return 0;
}

static void state_free(state *st)
{
    pipeline_free(st->clips, st->n_clips);
    tok_free(st->tok);
    cml_free(st->score);
    cml_free(st->select);
    free(st->texts);
    sentences_free(st->sent, st->n_sent);
    words_free(st->words, st->n_words);
    free(st->pcm);
}

int main(int argc, char **argv)
{
    args a;
    paths p;
    state st;
    if (parse_args(argc, argv, &a)) return usage();
    if (join(p.model, a.models, "clips.mlmodelc") || join(p.vocab, a.models, "clip_tokenizer.bin")
        || join(p.whisper, a.models, "ggml-base.en.bin")) {
        fprintf(stderr, "clips: models path too long\n");
        return 1;
    }
    memset(&st, 0, sizeof st);
    int rc = run(&a, &p, &st);
    if (!rc && a.json) print_json(&a, &st);
    if (!rc && !a.json) print_text(&st);
    state_free(&st);
    return rc;
}
