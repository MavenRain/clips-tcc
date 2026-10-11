/* Stage 2 gate: SPEC 1.5 vectors over synthetic RDTK containers, the real vocab, edge cases, speed. */
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOCAB "models/clip_tokenizer.bin"
#define TALK "fixtures/talk.txt"
#define META "\xE2\x96\x81" /* U+2581 */

static int passes, fails;

static void check(int ok, const char *what) {
    if (ok)
        passes++;
    else
        fails++;
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

static void show(const char *label, const int32_t *ids, int n) {
    printf("     %s (%d):", label, n);
    for (int i = 0; i < n && i < 16; i++)
        printf(" %d", ids[i]);
    printf("%s\n", n > 16 ? " ..." : "");
}

/* encode(text, max_len) == want[0..wn); prints the ids on a mismatch. */
static void expect_ids(const tok *t, const char *text, int max_len, const int32_t *want, int wn, const char *what) {
    int32_t got[256];
    int n = tok_encode(t, text, max_len, got);
    int ok = n == wn && memcmp(got, want, (size_t)wn * sizeof *got) == 0;
    check(ok, what);
    if (!ok)
        show("got", got, n);
}

static void put32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++)
        p[i] = (uint8_t)(v >> (8 * i));
}

/* ClipTests.swift vocabBytes(pieces:scores:): "RDTK", version 1, unk 3, bos 0, eos 2, count, f32 scores,
 * u16 lengths, piece bytes. Caller frees. */
static uint8_t *vocab_bytes(const char *const *pieces, const float *scores, int count, size_t *len) {
    size_t total = 21 + 6 * (size_t)count;
    for (int i = 0; i < count; i++)
        total += strlen(pieces[i]);
    uint8_t *b = malloc(total);
    if (!b)
        return NULL;
    memcpy(b, "RDTK\x01", 5);
    put32(b + 5, 3);
    put32(b + 9, 0);
    put32(b + 13, 2);
    put32(b + 17, (uint32_t)count);
    size_t at = 21;
    for (int i = 0; i < count; i++, at += 4) {
        uint32_t bits;
        memcpy(&bits, &scores[i], sizeof bits);
        put32(b + at, bits);
    }
    for (int i = 0; i < count; i++, at += 2) {
        size_t l = strlen(pieces[i]);
        b[at] = (uint8_t)(l & 0xFF);
        b[at + 1] = (uint8_t)((l >> 8) & 0xFF);
    }
    for (int i = 0; i < count; i++) {
        size_t l = strlen(pieces[i]);
        memcpy(b + at, pieces[i], l);
        at += l;
    }
    *len = total;
    return b;
}

static tok *from_pieces(const char *const *pieces, const float *scores, int count) {
    size_t n = 0;
    uint8_t *b = vocab_bytes(pieces, scores, count, &n);
    tok *t = b ? tok_from_bytes(b, n) : NULL;
    free(b);
    return t;
}

static const char *const SYN[] = {"<s>", "<pad>", "</s>", "<unk>", META "a", META "b", META "c"};
static const float SYN_SC[] = {0, 0, 0, 0, -1, -1, -1};

/* Loads a mutated copy of the synthetic container; 1 when it is rejected. */
static int rejects(void (*mutate)(uint8_t *b, size_t *n)) {
    size_t n = 0;
    uint8_t *b = vocab_bytes(SYN, SYN_SC, 7, &n);
    if (!b)
        return 0;
    uint8_t *big = realloc(b, n + 1);
    if (!big) {
        free(b);
        return 0;
    }
    mutate(big, &n);
    tok *t = tok_from_bytes(big, n);
    free(big);
    tok_free(t);
    return t == NULL;
}

static void m_magic(uint8_t *b, size_t *n) { b[3] = 'X'; }
static void m_count0(uint8_t *b, size_t *n) { put32(b + 17, 0); }
static void m_count_big(uint8_t *b, size_t *n) { put32(b + 17, (uint32_t)((*n - 21) / 6 + 1)); }
static void m_count_neg(uint8_t *b, size_t *n) { put32(b + 17, 0xFFFFFFFFu); }
static void m_unk(uint8_t *b, size_t *n) { put32(b + 5, 7); }
static void m_bos(uint8_t *b, size_t *n) { put32(b + 9, 0xFFFFFFFFu); }
static void m_eos(uint8_t *b, size_t *n) { put32(b + 13, 9); }
static void m_trailing(uint8_t *b, size_t *n) { b[(*n)++] = 'a'; }
static void m_short(uint8_t *b, size_t *n) { *n -= 1; }
static void m_len_past(uint8_t *b, size_t *n) { b[21 + 4 * 7 + 2 * 6] = 200; }
static void m_tiny(uint8_t *b, size_t *n) { *n = 20; }
/* Piece 6 "▁c" becomes "▁b": same total size, duplicate bytes. */
static void m_dup(uint8_t *b, size_t *n) { b[*n - 1] = 'b'; }
static void m_ok(uint8_t *b, size_t *n) { (void)b; (void)n; }

static void synthetic(void) {
    tok *t = from_pieces(SYN, SYN_SC, 7);
    check(t != NULL, "synthetic vocab loads");
    if (!t)
        return;

    /* :166-173 */
    char long_text[128] = "";
    for (int i = 0; i < 20; i++)
        strcat(long_text, "a b c ");
    int32_t ids[256];
    int n = tok_encode(t, long_text, 8, ids);
    check(n == 8 && ids[0] == 0 && ids[7] == 2, ":166 truncation to 8 keeps bos and eos");

    /* :175-181 */
    static const int32_t ab[] = {0, 4, 5, 2};
    expect_ids(t, "a b", 128, ab, 4, ":175 \"a b\" -> <s> \xE2\x96\x81" "a \xE2\x96\x81" "b </s>");
    expect_ids(t, "  a   b  ", 128, ab, 4, "\"  a   b  \" == \"a b\"");
    expect_ids(t, "a\xC2\xA0" "b", 128, ab, 4, "NBSP folds to U+0020 under NFKC");

    /* Each unknown scalar is its own unk; tabs are not squeezed. */
    static const int32_t tab[] = {0, 4, 3, 3, 2};
    expect_ids(t, "a\tb", 128, tab, 5, "\"a\\tb\" -> [0, 4, 3, 3, 2]");

    static const int32_t empty[] = {0, 3, 2};
    expect_ids(t, "", 128, empty, 3, "\"\" -> [bos, unk(U+2581 absent), eos]");
    expect_ids(t, "   ", 128, empty, 3, "all spaces == \"\"");

    static const int32_t one[] = {2};
    static const int32_t two[] = {0, 2};
    expect_ids(t, "a b", 1, one, 1, "max_len 1 -> [eos]");
    expect_ids(t, "a b", 2, two, 2, "max_len 2 -> [bos, eos]");
    check(tok_encode(t, "a b", 0, ids) == 0, "max_len 0 -> 0");
    check(tok_encode(t, "a b", -3, ids) == 0, "max_len < 0 -> 0");
    tok_free(t);

    /* :183-186 */
    static const uint8_t five[] = {0x00, 0x01, 0x02, 0x03, 0x04};
    tok *bad = tok_from_bytes(five, 5);
    check(bad == NULL, ":184 5-byte container rejected");
    tok_free(bad);
    size_t vn = 0;
    uint8_t *vb = vocab_bytes(SYN, SYN_SC, 7, &vn);
    bad = vb ? tok_from_bytes(vb, vn - 4) : NULL;
    check(vb && bad == NULL, ":185 synthetic minus last 4 bytes rejected");
    tok_free(bad);
    free(vb);

    /* SPEC 1.1 reject rules. */
    check(!rejects(m_ok), "unmutated copy loads");
    check(rejects(m_magic), "reject: bad magic");
    check(rejects(m_count0), "reject: count 0");
    check(rejects(m_count_big), "reject: count > (size - 21) / 6");
    check(rejects(m_count_neg), "reject: count -1");
    check(rejects(m_unk), "reject: unk out of range");
    check(rejects(m_bos), "reject: bos -1");
    check(rejects(m_eos), "reject: eos out of range");
    check(rejects(m_trailing), "reject: bytes after the last piece");
    check(rejects(m_short), "reject: last piece runs past the end");
    check(rejects(m_len_past), "reject: a length runs past the end");
    check(rejects(m_tiny), "reject: size < 21");
    check(rejects(m_dup), "reject: duplicate piece bytes");
}

static void kannada(void) {
    /* :194-209: canonically equal, byte-distinct pieces both load. */
    static const char *const P[] = {"<s>", "<pad>", "</s>", "<unk>", META "\xE0\xB3\x8B", META "\xE0\xB3\x8A\xE0\xB3\x95"};
    static const float S[] = {0, 0, 0, 0, -1, -1};
    tok *t = from_pieces(P, S, 6);
    check(t != NULL, ":206 colliding vocab loads");
    if (!t)
        return;
    static const int32_t want[] = {0, 4, 2};
    expect_ids(t, "\xE0\xB3\x8B", 128, want, 3, ":209 encode(U+0CCB) == [0, 4, 2]");
    expect_ids(t, "\xE0\xB3\x8A\xE0\xB3\x95", 128, want, 3, "U+0CCA U+0CD5 composes to U+0CCB -> [0, 4, 2]");
    tok_free(t);

    /* SPEC 1.3 ties: "▁ab" (-2) vs "▁a" + "b" (-2) keeps the longer piece. */
    static const char *const Q[] = {"<s>", "<pad>", "</s>", "<unk>", META "a", "b", META "ab"};
    static const float QS[] = {0, 0, 0, 0, -1, -1, -2};
    t = from_pieces(Q, QS, 7);
    static const int32_t tie[] = {0, 6, 2};
    if (t)
        expect_ids(t, "ab", 128, tie, 3, "tie keeps the longer piece (strict >)");
    else
        check(0, "tie vocab loads");
    tok_free(t);
}

/* Raw view of the real container, for piece bytes by id. */
typedef struct {
    uint8_t *buf;
    size_t n;
    int32_t count, unk, bos, eos;
    size_t *off;
} raw;

static int32_t rd32(const uint8_t *p) {
    uint32_t u = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    int32_t v;
    memcpy(&v, &u, sizeof v);
    return v;
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0)
        n = ftell(f);
    uint8_t *b = n >= 0 && fseek(f, 0, SEEK_SET) == 0 ? malloc((size_t)n + 1) : NULL;
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    if (b) {
        b[n] = 0;
        *len = (size_t)n;
    }
    return b;
}

static int raw_load(raw *r, const char *path) {
    memset(r, 0, sizeof *r);
    r->buf = read_file(path, &r->n);
    if (!r->buf || r->n < 21)
        return -1;
    r->unk = rd32(r->buf + 5);
    r->bos = rd32(r->buf + 9);
    r->eos = rd32(r->buf + 13);
    r->count = rd32(r->buf + 17);
    if (r->count <= 0 || (size_t)r->count > (r->n - 21) / 6)
        return -1;
    size_t c = (size_t)r->count;
    r->off = malloc((c + 1) * sizeof *r->off);
    if (!r->off)
        return -1;
    const uint8_t *ln = r->buf + 21 + 4 * c;
    r->off[0] = 21 + 6 * c;
    for (size_t i = 0; i < c; i++)
        r->off[i + 1] = r->off[i] + ((size_t)ln[2 * i] | (size_t)ln[2 * i + 1] << 8);
    return r->off[c] == r->n ? 0 : -1;
}

static void raw_free(raw *r) {
    free(r->buf);
    free(r->off);
}

static int piece_is(const raw *r, int32_t id, const char *s) {
    size_t l = strlen(s);
    return id >= 0 && id < r->count && r->off[id + 1] - r->off[id] == l && memcmp(r->buf + r->off[id], s, l) == 0;
}

static int32_t piece_id(const raw *r, const char *s) {
    for (int32_t i = 0; i < r->count; i++)
        if (piece_is(r, i, s))
            return i;
    return -1;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void speed(const tok *t, int32_t count) {
    size_t len = 0;
    uint8_t *txt = read_file(TALK, &len);
    check(txt != NULL, "fixtures/talk.txt reads");
    if (!txt)
        return;
    int lines = 0, bad = 0;
    int32_t ids[256];
    double t0 = now();
    for (char *line = (char *)txt; line; lines++) {
        char *nl = strchr(line, '\n');
        if (nl)
            *nl = 0;
        for (int r = 0; r < 100; r++) {
            int n = tok_encode(t, line, 256, ids);
            bad += n < 2 || ids[0] != 0 || ids[n - 1] != 2;
            for (int i = 0; i < n; i++)
                bad += ids[i] < 0 || ids[i] >= count;
        }
        line = nl ? nl + 1 : NULL;
    }
    double dt = now() - t0;
    printf("     talk.txt: %d lines x 100 encodes in %.3f s\n", lines, dt);
    check(bad == 0, "talk.txt encodes are well formed");
    check(dt < 2.0, "talk.txt x 100 under 2 s");
    free(txt);
}

static void real_vocab(void) {
    raw r;
    int rok = raw_load(&r, VOCAB) == 0;
    check(rok, "raw parse of " VOCAB);
    check(rok && r.count == 250002 && r.unk == 3 && r.bos == 0 && r.eos == 2, "count 250002, unk 3, bos 0, eos 2");
    tok *t = tok_load(VOCAB);
    check(t != NULL, "tok_load(" VOCAB ")");
    if (!rok || !t) {
        tok_free(t);
        raw_free(&r);
        return;
    }

    static const int32_t hello[] = {0, 35378, 8999, 38, 2};
    check(piece_is(&r, 35378, META "Hello") && piece_is(&r, 8999, META "world") && piece_is(&r, 38, "!"),
          "ids 35378, 8999, 38 are \xE2\x96\x81Hello, \xE2\x96\x81world, !");
    expect_ids(t, "Hello world!", 128, hello, 5, "\"Hello world!\" -> [0, 35378, 8999, 38, 2]");

    /* The real pair has no U+2581 prefix (the ClipTests vocab adds one), so U+0CCB encodes as
     * U+2581 (6) then U+0CCB (9440); the decomposed piece 80321 stays reachable only by exact bytes. */
    check(piece_is(&r, 9440, "\xE0\xB3\x8B") && piece_is(&r, 80321, "\xE0\xB3\x8A\xE0\xB3\x95") &&
              piece_id(&r, META "\xE0\xB3\x8B") == -1,
          "ids 9440 / 80321 are U+0CCB / U+0CCA U+0CD5; no \xE2\x96\x81 + U+0CCB piece");
    static const int32_t oo[] = {0, 6, 9440, 2};
    expect_ids(t, "\xE0\xB3\x8B", 128, oo, 4, "encode(U+0CCB) == [0, 6, 9440, 2]");
    expect_ids(t, "\xE0\xB3\x8A\xE0\xB3\x95", 128, oo, 4, "encode(U+0CCA U+0CD5) == [0, 6, 9440, 2] (NFKC composes)");

    /* 300 words at max_len 64. */
    static const char *const W[] = {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dogs"};
    char words[4096] = "";
    for (int i = 0; i < 300; i++) {
        strcat(words, W[i % 8]);
        strcat(words, i < 299 ? " " : "");
    }
    int32_t ids[256];
    int n = tok_encode(t, words, 64, ids);
    check(n == 64 && ids[0] == 0 && ids[63] == 2, "300 words at max_len 64 -> 64 ids, first 0, last 2");
    check(tok_encode(t, words, 0, ids) == 0, "max_len 0 -> 0");

    int32_t meta = piece_id(&r, META);
    int32_t empty[] = {0, meta, 2};
    printf("     id(\xE2\x96\x81) = %d\n", meta);
    check(meta >= 0, "U+2581 is a piece");
    expect_ids(t, "", 128, empty, 3, "\"\" -> [0, id(\xE2\x96\x81), 2]");

    int32_t a[256], b[256];
    int na = tok_encode(t, "  a   b  ", 128, a);
    int nb = tok_encode(t, "a b", 128, b);
    check(na > 0 && na == nb && memcmp(a, b, (size_t)na * sizeof *a) == 0, "\"  a   b  \" == \"a b\"");

    speed(t, r.count);
    tok_free(t);
    raw_free(&r);
}

int main(void) {
    synthetic();
    kannada();
    real_vocab();
    printf("%d passed, %d failed\n", passes, fails);
    return fails ? 1 : 0;
}
