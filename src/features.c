/* The 5 discourse features of the selector (SPEC 2.1-2.5, Pipeline.swift:178-221).
 *
 * The cue patterns go through the SDK's Regex module. Its Apple backend
 * (Sources/Regex/FoundationRegex.swift) is NSRegularExpression with options [.caseInsensitive]
 * only, so: ICU default \b (no UREGEX_UWORD), and ICU case-insensitive literal strings, which
 * compare full case foldings. No regex library here: each pattern is a set of literal
 * alternatives with optional \b checks, matched over the lowercased, case-folded scalars. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utf8proc.h>

#include "features.h"
#include "uni.h"

typedef struct {
    int lead, trail;        /* \b before / after the literal */
    const char *alts[8];    /* ASCII alternatives, NULL-terminated; "'?" is expanded */
} cue;

/* hookPatterns, Pipeline.swift:182-188. */
static const cue hooks[] = {
    {1, 1, {"here's", "heres", NULL}},
    {0, 0, {"nobody", NULL}},
    {0, 0, {"most people", NULL}},
    {0, 0, {"the mistake", "the secret", "the truth", "the problem", "the reason", NULL}},
    {1, 1, {"what if", NULL}},
    {1, 1, {"why", NULL}},
    {1, 1, {"how to", "how i", "how we", "how you", NULL}},
    {0, 0, {"?", NULL}},
    {1, 1, {"never", "always", "everyone", "no one", "nobody", NULL}},
    {1, 1, {"biggest", "worst", "best", "craziest", "first", "only", "most", NULL}},
};

/* payoffPatterns, Pipeline.swift:189-191. */
static const cue payoffs[] = {
    {1, 1, {"so", NULL}},
    {0, 0, {"that's why", "thats why", NULL}},
    {0, 0, {"the point", NULL}},
    {0, 0, {"which means", NULL}},
    {0, 0, {"turns out", NULL}},
    {0, 0, {"the result", NULL}},
};

#define CAT(x) (1UL << UTF8PROC_CATEGORY_##x)

/* ICU \w: [\p{Alphabetic}\p{M}\p{Nd}\p{Pc}‌‍]; Alphabetic = L* + Nl + Other_Alphabetic. */
static const unsigned long word_cats = CAT(LU) | CAT(LL) | CAT(LT) | CAT(LM) | CAT(LO) | CAT(NL)
                                     | CAT(MN) | CAT(MC) | CAT(ME) | CAT(ND) | CAT(PC);

/* Swift Character.isNumber: the first scalar has a Numeric_Type. utf8proc has no Unihan data, so
 * CJK ideographs with kPrimaryNumeric (U+4E00 and friends) are missed here. */
static const unsigned long numeric_cats = CAT(ND) | CAT(NL) | CAT(NO);

/* Lowercased scalars and their full case foldings, unit by unit. */
typedef struct {
    int32_t *cp;        /* lowercased scalars */
    long n;
    int32_t *unit;      /* case-folded units, concatenated */
    long *owner;        /* unit -> scalar index */
    long *first;        /* scalar -> its first unit; first[n] = nu */
    long nu;
} folded;

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

static int in_cats(unsigned long cats, int32_t c)
{
    return (int)((cats >> utf8proc_category(c)) & 1UL);
}

static void fold_free(folded *f)
{
    free(f->cp);
    free(f->unit);
    free(f->owner);
    free(f->first);
}

/* Lowercase per scalar and retain the U+0130 SpecialCasing expansion. The combining dot
 * prevents an ICU word boundary immediately after the resulting i. */
static int fold_text(const char *s, size_t len, folded *f)
{
    size_t i = 0;
    memset(f, 0, sizeof *f);
    f->cp = malloc((len + 1) * sizeof *f->cp);
    f->unit = malloc((4 * len + 1) * sizeof *f->unit);
    f->owner = malloc((4 * len + 1) * sizeof *f->owner);
    f->first = malloc((len + 1) * sizeof *f->first);
    if (!f->cp || !f->unit || !f->owner || !f->first) {
        fold_free(f);
        return -1;
    }
    while (i < len) {
        int32_t source = next_cp(s, len, &i);
        int32_t lower[2] = {utf8proc_tolower(source), 0};
        int nlower = 1;
        if (source == 0x0130) {
            lower[0] = 'i';
            lower[1] = 0x0307;
            nlower = 2;
        }
        for (int m = 0; m < nlower; m++) {
            int32_t c = lower[m];
            utf8proc_int32_t buf[4];
            int last = 0;
            utf8proc_ssize_t k = utf8proc_decompose_char(c, buf, 4, UTF8PROC_CASEFOLD, &last);
            if (k < 1 || k > 4) {
                buf[0] = c;
                k = 1;
            }
            f->cp[f->n] = c;
            f->first[f->n] = f->nu;
            for (utf8proc_ssize_t j = 0; j < k; j++) {
                f->unit[f->nu] = buf[j];
                f->owner[f->nu] = f->n;
                f->nu++;
            }
            f->n++;
        }
    }
    f->first[f->n] = f->nu;
    return 0;
}

static int is_word(int32_t c)
{
    /* Other_Alphabetic outside \p{M}: circled and squared Latin letters (So). */
    return in_cats(word_cats, c) || c == 0x200C || c == 0x200D
        || (c >= 0x24B6 && c <= 0x24E9) || (c >= 0x1F130 && c <= 0x1F149)
        || (c >= 0x1F150 && c <= 0x1F169) || (c >= 0x1F170 && c <= 0x1F189);
}

/* ICU isWordBoundary skips Grapheme_Extend and Cf scalars. utf8proc's GCB Extend class is
 * Grapheme_Extend plus the emoji modifiers U+1F3FB..U+1F3FF, which are taken back out. */
static int ignorable(int32_t c)
{
    const utf8proc_property_t *p = utf8proc_get_property(c);
    int extend = p->boundclass == UTF8PROC_BOUNDCLASS_EXTEND && !(c >= 0x1F3FB && c <= 0x1F3FF);
    return extend || p->category == UTF8PROC_CATEGORY_CF;
}

/* ICU RegexMatcher::isWordBoundary before scalar index pos: never before an ignorable scalar;
 * else word(scalar at pos) != word(nearest non-ignorable scalar before pos), outside = non-word. */
static int boundary(const folded *f, long pos)
{
    int cur = 0, prev = 0;
    long i = pos;
    if (pos < f->n) {
        if (ignorable(f->cp[pos]))
            return 0;
        cur = is_word(f->cp[pos]);
    }
    while (i > 0 && ignorable(f->cp[i - 1]))
        i--;
    if (i > 0)
        prev = is_word(f->cp[i - 1]);
    return cur != prev;
}

/* alt matches the folded units from scalar s on, ending on a scalar edge (ICU URX_STRING_I). */
static int alt_at(const folded *f, long s, const char *alt, int lead, int trail)
{
    long u = f->first[s], m = (long)strlen(alt), end;
    if (m == 0 || u + m > f->nu)
        return 0;
    for (long j = 0; j < m; j++)
        if (f->unit[u + j] != (int32_t)(unsigned char)alt[j])
            return 0;
    end = f->owner[u + m - 1] + 1;
    if (f->first[end] != u + m)
        return 0;
    if (lead && !boundary(f, s))
        return 0;
    return !trail || boundary(f, end);
}

static int cue_matches(const folded *f, const cue *c)
{
    for (long s = 0; s < f->n; s++)
        for (int a = 0; c->alts[a]; a++)
            if (alt_at(f, s, c->alts[a], c->lead, c->trail))
                return 1;
    return 0;
}

static int count_cues(const folded *f, const cue *cs, size_t n)
{
    int count = 0;
    for (size_t i = 0; i < n; i++)
        count += cue_matches(f, &cs[i]);
    return count;
}

/* Clusters of t whose first scalar is numeric (Character.isNumber). */
static long numeric_clusters(const char *t, size_t len)
{
    long count = 0;
    utf8proc_int32_t state = 0, prev = -1;
    size_t i = 0;
    while (i < len) {
        int32_t cp = next_cp(t, len, &i);
        int starts = prev < 0 || utf8proc_grapheme_break_stateful(prev, cp, &state);
        count += starts && in_cats(numeric_cats, cp);
        prev = cp;
    }
    return count;
}

/* trimmedWhitespace.hasSuffix("?"): Swift compares Characters, so the last cluster is "?". */
static int ends_question(const char *t, size_t len)
{
    size_t b = 0, e = 0, last;
    uni_trim(t, len, &b, &e);
    if (e <= b)
        return 0;
    last = b + uni_last_grapheme(t + b, e - b);
    return e - last == 1 && t[last] == '?';
}

/* A U+0020 separates only when it is a whole grapheme cluster: Swift split(separator: " ")
 * compares Characters, and " " + a combining mark is one Character. */
static int lone_space(const char *s, size_t len, size_t i)
{
    size_t k;
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

static int is_sep(const char *s, size_t len, size_t i)
{
    return s[i] == ' ' && lone_space(s, len, i);
}

long features_word_count(const char *s)
{
    size_t len = s ? strlen(s) : 0, i = 0;
    long count = 0;
    while (i < len) {
        while (i < len && is_sep(s, len, i))
            i++;
        if (i >= len)
            break;
        count++;
        while (i < len && !is_sep(s, len, i))
            i++;
    }
    return count;
}

static char *encode(const folded *f, size_t *len)
{
    char *t = malloc(4 * (size_t)f->n + 1);
    size_t at = 0;
    if (!t)
        return NULL;
    for (long i = 0; i < f->n; i++)
        at += (size_t)utf8proc_encode_char(f->cp[i], (utf8proc_uint8_t *)t + at);
    t[at] = '\0';
    *len = at;
    return t;
}

int features_disc(const char *sentence, long index, long n, float out[5])
{
    folded f;
    size_t tl = 0;
    char *t;
    long digits;
    if (!sentence || !out) {
        fprintf(stderr, "features: no sentence\n");
        return -1;
    }
    if (fold_text(sentence, strlen(sentence), &f) != 0) {
        fprintf(stderr, "features: out of memory\n");
        return -1;
    }
    t = encode(&f, &tl);
    if (!t) {
        fold_free(&f);
        fprintf(stderr, "features: out of memory\n");
        return -1;
    }
    digits = numeric_clusters(t, tl);
    out[0] = (float)((double)index / (double)(n > 0 ? n : 1));
    out[1] = (float)count_cues(&f, hooks, sizeof hooks / sizeof hooks[0]);
    out[2] = (float)count_cues(&f, payoffs, sizeof payoffs / sizeof payoffs[0]);
    out[3] = (float)ends_question(t, tl);
    out[4] = (float)digits / (float)(uni_graphemes(t, tl) + 1);
    free(t);
    fold_free(&f);
    return 0;
}
