/* Shared Unicode helpers over utf8proc. */
#include "uni.h"

#include <utf8proc.h>

int uni_is_space(int32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

/* Decode one scalar at p; an invalid sequence reads as U+FFFD over one byte. */
static size_t next(const char *p, size_t left, int32_t *cp) {
    utf8proc_ssize_t k = utf8proc_iterate((const utf8proc_uint8_t *)p, (utf8proc_ssize_t)left, cp);
    if (k > 0) return (size_t)k;
    *cp = 0xFFFD;
    return 1;
}

/* Walk the clusters of s; return the count and set *last to the start of the last one. */
static long walk(const char *s, size_t len, size_t *last) {
    int32_t prev = -1, cp, state = 0;
    long n = 0;
    *last = len;
    for (size_t i = 0, k; i < len; i += k, prev = cp) {
        k = next(s + i, len - i, &cp);
        if (prev < 0 || utf8proc_grapheme_break_stateful(prev, cp, &state)) {
            n++;
            *last = i;
        }
    }
    return n;
}

long uni_graphemes(const char *s, size_t len) {
    size_t last;
    return walk(s, len, &last);
}

size_t uni_last_grapheme(const char *s, size_t len) {
    size_t last;
    walk(s, len, &last);
    return last;
}

void uni_trim(const char *s, size_t len, size_t *b, size_t *e) {
    int32_t cp;
    int seen = 0;
    *b = *e = 0;
    for (size_t i = 0, k; i < len; i += k) {
        k = next(s + i, len - i, &cp);
        if (uni_is_space(cp)) continue;
        if (!seen) *b = i;
        seen = 1;
        *e = i + k;
    }
    if (!seen) *b = *e = len;
}
