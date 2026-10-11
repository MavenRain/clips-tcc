/* XLM-R SentencePiece Unigram over the RDTK container (SPEC section 1).
 * The tok is read-only after load; tok_encode keeps all scratch on the heap per call. */
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utf8proc.h>

#define TOK_HDR 21        /* magic 4, version 1, unk/bos/eos/count 4 each */
#define TOK_MAX_PIECE 32  /* maxLen cap in scalars */
#define TOK_SPACE 0x20
#define TOK_META 0x2581   /* U+2581 LOWER ONE EIGHTH BLOCK */

struct tok {
    uint8_t *buf;          /* owned container bytes */
    const uint8_t *pieces; /* piece bytes back to back, id order */
    uint32_t *bounds;      /* count + 1 offsets into pieces */
    float *scores;
    int32_t *slots;        /* open addressing over exact piece bytes, -1 = empty */
    size_t mask;
    int32_t count, unk, bos, eos;
    int max_len;
    double unk_penalty;
};

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int32_t rd_i32(const uint8_t *p) {
    uint32_t u = rd_u32(p);
    int32_t v;
    memcpy(&v, &u, sizeof v);
    return v;
}

static float rd_f32(const uint8_t *p) {
    uint32_t u = rd_u32(p);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* FNV-1a 64; lookups still compare the exact bytes. */
static uint64_t fnv(const uint8_t *p, size_t n) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}

static size_t piece_len(const tok *t, int32_t id) {
    return t->bounds[id + 1] - t->bounds[id];
}

/* Slot that holds the piece equal to p[0..n), or the empty slot where it would go. */
static size_t probe(const tok *t, const uint8_t *p, size_t n) {
    size_t s = (size_t)fnv(p, n) & t->mask;
    while (t->slots[s] != -1) {
        int32_t id = t->slots[s];
        if (piece_len(t, id) == n && memcmp(t->pieces + t->bounds[id], p, n) == 0)
            return s;
        s = (s + 1) & t->mask;
    }
    return s;
}

static int32_t lookup(const tok *t, const uint8_t *p, size_t n) {
    return t->slots[probe(t, p, n)];
}

/* Number of UTF-8 lead bytes. */
static int scalar_count(const uint8_t *p, size_t n) {
    int c = 0;
    for (size_t i = 0; i < n; i++)
        c += (p[i] & 0xC0) != 0x80;
    return c;
}

void tok_free(tok *t) {
    if (!t)
        return;
    free(t->buf);
    free(t->bounds);
    free(t->scores);
    free(t->slots);
    free(t);
}

static tok *reject(tok *t, const char *why) {
    fprintf(stderr, "tokenizer: %s\n", why);
    tok_free(t);
    return NULL;
}

/* Exact-bytes index; -1 out of memory, -2 duplicate piece bytes. */
static int build_index(tok *t) {
    size_t cap = 16;
    while (cap < 2 * (size_t)t->count)
        cap <<= 1;
    t->slots = malloc(cap * sizeof *t->slots);
    if (!t->slots)
        return -1;
    memset(t->slots, 0xFF, cap * sizeof *t->slots);
    t->mask = cap - 1;
    for (int32_t id = 0; id < t->count; id++) {
        size_t s = probe(t, t->pieces + t->bounds[id], piece_len(t, id));
        if (t->slots[s] != -1)
            return -2;
        t->slots[s] = id;
    }
    return 0;
}

static int id_ok(const tok *t, int32_t id) {
    return id >= 0 && id < t->count;
}

/* Takes ownership of buf on every path. */
static tok *build(uint8_t *buf, size_t n) {
    tok *t = calloc(1, sizeof *t);
    if (!t) {
        free(buf);
        fprintf(stderr, "tokenizer: out of memory\n");
        return NULL;
    }
    t->buf = buf;
    if (n < TOK_HDR || n > INT32_MAX)
        return reject(t, "bad container size");
    if (memcmp(buf, "RDTK", 4) != 0)
        return reject(t, "bad magic");
    t->unk = rd_i32(buf + 5);
    t->bos = rd_i32(buf + 9);
    t->eos = rd_i32(buf + 13);
    t->count = rd_i32(buf + 17);
    if (t->count <= 0 || (size_t)t->count > (n - TOK_HDR) / 6)
        return reject(t, "bad piece count");

    size_t c = (size_t)t->count;
    const uint8_t *sc = buf + TOK_HDR;
    const uint8_t *ln = sc + 4 * c;
    size_t base = TOK_HDR + 6 * c;
    t->scores = malloc(c * sizeof *t->scores);
    t->bounds = malloc((c + 1) * sizeof *t->bounds);
    if (!t->scores || !t->bounds)
        return reject(t, "out of memory");
    t->pieces = buf + base;

    size_t pos = 0;
    int longest = 1;
    float lo = rd_f32(sc);
    for (size_t id = 0; id < c; id++) {
        size_t len = (size_t)ln[2 * id] | (size_t)ln[2 * id + 1] << 8;
        float s = rd_f32(sc + 4 * id);
        if (!isfinite(s))
            return reject(t, "non-finite piece score");
        if (len > n - base - pos)
            return reject(t, "piece runs past the end");
        t->scores[id] = s;
        lo = s < lo ? s : lo;
        t->bounds[id] = (uint32_t)pos;
        int k = scalar_count(t->pieces + pos, len);
        longest = k > longest ? k : longest;
        pos += len;
    }
    t->bounds[c] = (uint32_t)pos;
    if (base + pos != n)
        return reject(t, "pieces do not end at EOF");
    if (!id_ok(t, t->unk) || !id_ok(t, t->bos) || !id_ok(t, t->eos))
        return reject(t, "special id out of range");
    t->max_len = longest < TOK_MAX_PIECE ? longest : TOK_MAX_PIECE;
    t->unk_penalty = (double)lo - 10.0;

    int r = build_index(t);
    if (r == -1)
        return reject(t, "out of memory");
    if (r == -2)
        return reject(t, "duplicate piece bytes");
    return t;
}

tok *tok_from_bytes(const uint8_t *bytes, size_t n) {
    if (!bytes && n)
        return NULL;
    uint8_t *buf = malloc(n ? n : 1);
    if (!buf) {
        fprintf(stderr, "tokenizer: out of memory\n");
        return NULL;
    }
    if (n)
        memcpy(buf, bytes, n);
    return build(buf, n);
}

tok *tok_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open %s\n", path);
        return NULL;
    }
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0)
        n = ftell(f);
    if (n < 0 || n > INT32_MAX || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "tokenizer: bad file %s\n", path);
        return NULL;
    }
    uint8_t *buf = malloc(n ? (size_t)n : 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        fprintf(stderr, "tokenizer: cannot read %s\n", path);
        return NULL;
    }
    fclose(f);
    return build(buf, (size_t)n);
}

/* Per-call scratch, all sized from the NFKC byte length. */
typedef struct {
    int32_t *cps;   /* normalized scalars, U+2581 prefix and spaces */
    uint8_t *bytes; /* their UTF-8 */
    size_t *span;   /* byte offset of scalar i, n + 1 entries */
    double *best;
    int32_t *bpos;
    int32_t *bid;
} scratch;

static void scratch_free(scratch *s) {
    free(s->cps);
    free(s->bytes);
    free(s->span);
    free(s->best);
    free(s->bpos);
    free(s->bid);
}

static int scratch_alloc(scratch *s, size_t cap) {
    s->cps = malloc(cap * sizeof *s->cps);
    s->bytes = malloc(4 * cap);
    s->span = malloc((cap + 1) * sizeof *s->span);
    s->best = malloc((cap + 1) * sizeof *s->best);
    s->bpos = malloc((cap + 1) * sizeof *s->bpos);
    s->bid = malloc((cap + 1) * sizeof *s->bid);
    return s->cps && s->bytes && s->span && s->best && s->bpos && s->bid ? 0 : -1;
}

/* SPEC 1.2 steps 2-3 over NFKC bytes: squeeze U+0020 runs, drop leading and one trailing,
 * prefix U+2581 and map each kept space to it. Returns the scalar count or -1. */
static long squeeze(const uint8_t *nf, size_t nl, int32_t *cps) {
    size_t n = 0, pos = 0;
    int last_space = 1;
    cps[n++] = TOK_META;
    while (pos < nl) {
        utf8proc_int32_t cp;
        utf8proc_ssize_t adv = utf8proc_iterate(nf + pos, (utf8proc_ssize_t)(nl - pos), &cp);
        if (adv <= 0)
            return -1;
        pos += (size_t)adv;
        if (cp == TOK_SPACE && last_space)
            continue;
        cps[n++] = cp == TOK_SPACE ? TOK_META : cp;
        last_space = cp == TOK_SPACE;
    }
    if (n > 1 && last_space)
        n--;
    return (long)n;
}

/* SPEC 1.3. Fills bpos/bid for positions 1..n. */
static void viterbi(const tok *t, const scratch *s, size_t n) {
    s->best[0] = 0.0;
    s->bpos[0] = -1;
    s->bid[0] = -1;
    for (size_t i = 1; i <= n; i++) {
        double best = -HUGE_VAL; /* any finite unk step beats it, so no id -1 */
        int32_t bp = -1, bi = -1;
        size_t lo = i > (size_t)t->max_len ? i - (size_t)t->max_len : 0;
        for (size_t j = lo; j < i; j++) {
            int32_t id = lookup(t, s->bytes + s->span[j], s->span[i] - s->span[j]);
            if (id < 0)
                continue;
            double sc = s->best[j] + (double)t->scores[id];
            if (sc > best) { /* strict: ties keep the longer piece */
                best = sc;
                bp = (int32_t)j;
                bi = id;
            }
        }
        double cand = s->best[i - 1] + t->unk_penalty;
        if (cand > best) {
            best = cand;
            bp = (int32_t)(i - 1);
            bi = t->unk;
        }
        s->best[i] = best;
        s->bpos[i] = bp;
        s->bid[i] = bi;
    }
}

int tok_encode(const tok *t, const char *utf8, int max_len, int32_t *out) {
    if (max_len <= 0)
        return 0;
    if (!t || !utf8 || !out)
        return -1;

    utf8proc_uint8_t *nf = NULL;
    utf8proc_ssize_t nl = utf8proc_map((const utf8proc_uint8_t *)utf8, 0, &nf,
                                       UTF8PROC_NULLTERM | UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_COMPAT);
    if (nl < 0) {
        fprintf(stderr, "tokenizer: NFKC failed: %s\n", utf8proc_errmsg(nl));
        return -1;
    }

    scratch s = {0};
    if (scratch_alloc(&s, (size_t)nl + 1) != 0) {
        fprintf(stderr, "tokenizer: out of memory\n");
        scratch_free(&s);
        free(nf);
        return -1;
    }
    long sn = squeeze(nf, (size_t)nl, s.cps);
    free(nf);
    if (sn < 0) {
        fprintf(stderr, "tokenizer: bad NFKC output\n");
        scratch_free(&s);
        return -1;
    }

    size_t n = (size_t)sn, len = 0;
    for (size_t i = 0; i < n; i++) {
        s.span[i] = len;
        len += (size_t)utf8proc_encode_char(s.cps[i], s.bytes + len);
    }
    s.span[n] = len;
    viterbi(t, &s, n);

    /* Backtrack from n: count the steps, then fill seq = [bos] + pieces + [eos] back to front. */
    size_t k = 0;
    for (int32_t i = (int32_t)n; i > 0; i = s.bpos[i])
        k++;
    int32_t *seq = malloc((k + 2) * sizeof *seq);
    if (!seq) {
        fprintf(stderr, "tokenizer: out of memory\n");
        scratch_free(&s);
        return -1;
    }
    size_t at = k;
    for (int32_t i = (int32_t)n; i > 0; i = s.bpos[i])
        seq[at--] = s.bid[i];
    seq[0] = t->bos;
    seq[k + 1] = t->eos;
    scratch_free(&s);

    size_t total = k + 2;
    size_t m = total > (size_t)max_len ? (size_t)max_len : total;
    memcpy(out, seq, m * sizeof *out);
    if (total > m)
        out[m - 1] = t->eos;
    free(seq);
    return (int)m;
}
