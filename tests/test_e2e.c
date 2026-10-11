/* Stage 5 gate: build/clips on fixtures/talk.mp4 with --json. At most 8 spans, best first,
 * no two spans overlap in time or sentence ids, every start < end, 0 < durationSec <= end - start
 * (pauses over 0.3 s are cut), the durations sum to at most materialSec, input is the absolute path.
 * Run from the repo root, unsandboxed. */
#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CMD "build/clips fixtures/talk.mp4 --json 2>build/e2e.stderr"
#ifndef TEST_FFPROBE
#define TEST_FFPROBE "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe"
#endif
#define PROBE_CMD TEST_FFPROBE " -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 fixtures/talk.mp4"
#define EPS 1e-9 /* touching ranges share an end point */

enum { CAP = 64, MAX_SPANS = 8, MAX_DEPTH = 64, PATH_LEN = 4096 };
enum { F_START = 1, F_END = 2, F_SCORE = 4, F_ID = 8, F_IDS = 16, F_TEXT = 32, F_DUR = 64, F_ALL = 127 };

typedef struct { const char *p; } cur;
typedef struct { double start, end, score, id, lo, hi, dur; int n_ids, text_len; } span;

static int checks, failed;

static void check(int ok, const char *fmt, ...)
{
    va_list ap;
    checks++;
    if (ok) return;
    failed++;
    fprintf(stderr, "FAIL: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
}

/* Whole stdout of cmd, NUL-terminated in *out. Returns the pclose status, -1 on a local failure. */
static int capture(const char *cmd, char **out, size_t *len)
{
    FILE *f = popen(cmd, "r");
    size_t cap = 0, got;
    char chunk[4096];
    if (!f) return -1;
    while ((got = fread(chunk, 1, sizeof chunk, f)) > 0) {
        if (*len + got + 1 > cap) {
            size_t ncap = (*len + got + 1) * 2;
            char *nb = realloc(*out, ncap);
            if (!nb) {
                pclose(f);
                return -1;
            }
            *out = nb;
            cap = ncap;
        }
        memcpy(*out + *len, chunk, got);
        *len += got;
        (*out)[*len] = 0;
    }
    return pclose(f);
}

/* ---- minimal JSON reader ---- */

static void ws(cur *c)
{
    while (*c->p == ' ' || *c->p == '\n' || *c->p == '\r' || *c->p == '\t') c->p++;
}

static int eat(cur *c, char ch)
{
    ws(c);
    if (*c->p != ch) return 0;
    c->p++;
    return 1;
}

/* A string; its raw bytes (escapes kept) go to out when out != NULL. Sets *n to the raw length. */
static int str(cur *c, char *out, size_t cap, int *n)
{
    size_t k = 0;
    int raw = 0;
    if (!eat(c, '"')) return -1;
    for (; *c->p && *c->p != '"'; c->p++, raw++) {
        if (*c->p == '\\') {
            if (!c->p[1]) return -1;
            if (out && k + 1 < cap) out[k++] = *c->p;
            c->p++;
            raw++;
        }
        if (out && k + 1 < cap) out[k++] = *c->p;
    }
    if (out) out[k] = 0;
    if (n) *n = raw;
    if (*c->p != '"') return -1;
    c->p++;
    return 0;
}

static int num(cur *c, double *v)
{
    char *end = NULL;
    ws(c);
    *v = strtod(c->p, &end);
    if (end == c->p) return -1;
    c->p = end;
    return 0;
}

static int skip(cur *c, int depth);

static int skip_members(cur *c, int depth, char close, int keyed)
{
    if (eat(c, close)) return 0;
    do {
        if (keyed && (str(c, NULL, 0, NULL) || !eat(c, ':'))) return -1;
        if (skip(c, depth + 1)) return -1;
    } while (eat(c, ','));
    return eat(c, close) ? 0 : -1;
}

/* Any value. */
static int skip(cur *c, int depth)
{
    double v;
    if (depth > MAX_DEPTH) return -1;
    ws(c);
    if (*c->p == '"') return str(c, NULL, 0, NULL);
    if (eat(c, '{')) return skip_members(c, depth, '}', 1);
    if (eat(c, '[')) return skip_members(c, depth, ']', 0);
    if (!strncmp(c->p, "true", 4) || !strncmp(c->p, "null", 4)) {
        c->p += 4;
        return 0;
    }
    if (!strncmp(c->p, "false", 5)) {
        c->p += 5;
        return 0;
    }
    return num(c, &v);
}

/* sentenceIDs: their min, max and count. */
static int ids(cur *c, span *s)
{
    double v;
    s->lo = 1e18;
    s->hi = -1e18;
    s->n_ids = 0;
    if (!eat(c, '[')) return -1;
    if (eat(c, ']')) return 0;
    do {
        if (num(c, &v)) return -1;
        if (v < s->lo) s->lo = v;
        if (v > s->hi) s->hi = v;
        s->n_ids++;
    } while (eat(c, ','));
    return eat(c, ']') ? 0 : -1;
}

static int field(cur *c, span *s, const char *key, int *seen)
{
    if (!strcmp(key, "start")) return *seen |= F_START, num(c, &s->start);
    if (!strcmp(key, "end")) return *seen |= F_END, num(c, &s->end);
    if (!strcmp(key, "durationSec")) return *seen |= F_DUR, num(c, &s->dur);
    if (!strcmp(key, "score")) return *seen |= F_SCORE, num(c, &s->score);
    if (!strcmp(key, "id")) return *seen |= F_ID, num(c, &s->id);
    if (!strcmp(key, "sentenceIDs")) return *seen |= F_IDS, ids(c, s);
    if (!strcmp(key, "text")) return *seen |= F_TEXT, str(c, NULL, 0, &s->text_len);
    return skip(c, 1);
}

/* One Pick object; every field the gate checks must be present. */
static int clip(cur *c, span *s)
{
    int seen = 0;
    char key[32];
    memset(s, 0, sizeof *s);
    if (!eat(c, '{') || eat(c, '}')) return -1;
    do {
        if (str(c, key, sizeof key, NULL) || !eat(c, ':') || field(c, s, key, &seen)) return -1;
    } while (eat(c, ','));
    return eat(c, '}') && seen == F_ALL ? 0 : -1;
}

static int clips(cur *c, span *out, int cap, int *count)
{
    *count = 0;
    if (!eat(c, '[')) return -1;
    if (eat(c, ']')) return 0;
    do {
        if (*count >= cap || clip(c, &out[*count])) return -1;
        (*count)++;
    } while (eat(c, ','));
    return eat(c, ']') ? 0 : -1;
}

/* The ClipReport object; only "clips", "materialSec" and "input" are read, the rest is skipped. */
static int report(cur *c, span *out, int cap, int *count, double *material, char *input)
{
    char key[32];
    *count = -1;
    if (!eat(c, '{') || eat(c, '}')) return -1;
    do {
        if (str(c, key, sizeof key, NULL) || !eat(c, ':')) return -1;
        int r = !strcmp(key, "clips") ? clips(c, out, cap, count)
              : !strcmp(key, "materialSec") ? num(c, material)
              : !strcmp(key, "input") ? str(c, input, PATH_LEN, NULL) : skip(c, 1);
        if (r) return -1;
    } while (eat(c, ','));
    if (!eat(c, '}')) return -1;
    ws(c);
    return *c->p || *count < 0 ? -1 : 0;
}

/* ---- gate ---- */

static void check_span(const span *s, int i)
{
    check(s->start < s->end, "span %d: start %.3f < end %.3f", i + 1, s->start, s->end);
    check(s->start >= 0, "span %d: start %.3f >= 0", i + 1, s->start);
    check(s->id == i + 1, "span %d: id %.0f is its rank", i + 1, s->id);
    check(s->n_ids > 0 && s->lo <= s->hi, "span %d: has sentence ids", i + 1);
    check(s->text_len > 0, "span %d: has text", i + 1);
    check(s->dur > 0 && s->dur <= s->end - s->start + EPS, "span %d: durationSec %.3f in (0, end - start %.3f]", i + 1,
          s->dur, s->end - s->start);
}

static void check_pair(const span *a, const span *b, int i, int j)
{
    int time = a->start < b->end - EPS && b->start < a->end - EPS;
    int ids = a->lo <= b->hi && b->lo <= a->hi;
    check(!time, "spans %d [%.3f, %.3f] and %d [%.3f, %.3f] overlap in time", i + 1, a->start, a->end, j + 1,
          b->start, b->end);
    check(!ids, "spans %d [%.0f..%.0f] and %d [%.0f..%.0f] share sentence ids", i + 1, a->lo, a->hi, j + 1, b->lo,
          b->hi);
}

int main(void)
{
    char *buf = NULL;
    size_t len = 0;
    span s[CAP];
    int n = -1;
    double material = -1, total = 0;
    char input[PATH_LEN] = "", want[PATH_LEN];
    int status = capture(CMD, &buf, &len);
    cur c = { buf ? buf : "" };

    check(status == 0, "clips exits 0 (status %d, stderr in build/e2e.stderr)", status);
    check(report(&c, s, CAP, &n, &material, input) == 0, "--json output parses as a ClipReport with a clips array");
    /* Speech synthesis can change the fixture length. Probe the container independently. */
    char *meta = NULL;
    size_t meta_len = 0;
    double expected = -1;
    int probe_status = capture(PROBE_CMD, &meta, &meta_len);
    cur m = { meta ? meta : "" };
    int parsed = num(&m, &expected);
    ws(&m);
    check(probe_status == 0 && parsed == 0 && !*m.p && expected > 0 && isfinite(expected),
          "ffprobe supplies a positive container duration (status %d)", probe_status);
    check(material > expected - 0.01 && material < expected + 0.01,
          "materialSec %.3f matches the %.3f s container duration", material, expected);
    free(meta);
    check(realpath("fixtures/talk.mp4", want) && !strcmp(input, want), "input \"%s\" is the absolute path", input);
    check(n >= 1, "at least one span (got %d)", n);
    check(n <= MAX_SPANS, "at most %d spans (got %d)", MAX_SPANS, n);
    for (int i = 0; i < n; i++) check_span(&s[i], i);
    for (int i = 0; i < n; i++) total += s[i].dur;
    check(total <= material + EPS, "durationSec sum %.3f <= materialSec %.3f", total, material);
    for (int i = 1; i < n; i++)
        check(s[i - 1].score >= s[i].score, "score order: span %d %.4f >= span %d %.4f", i, s[i - 1].score, i + 1,
              s[i].score);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) check_pair(&s[i], &s[j], i, j);
    free(buf);
    printf("test_e2e: %d spans, %d checks, %d failed\n", n, checks, failed);
    return failed ? 1 : 0;
}
