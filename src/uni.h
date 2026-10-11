/* Shared Unicode helpers over utf8proc: White_Space, grapheme clusters, trimming. */
#ifndef CLIPS_UNI_H
#define CLIPS_UNI_H

#include <stddef.h>
#include <stdint.h>

/* The Unicode White_Space property (Swift Unicode.Scalar.Properties.isWhitespace). */
int uni_is_space(int32_t cp);

/* Number of extended grapheme clusters (Swift String.count). Invalid bytes count as U+FFFD. */
long uni_graphemes(const char *s, size_t len);

/* Byte offset where the last grapheme cluster starts; len when s is empty. */
size_t uni_last_grapheme(const char *s, size_t len);

/* [*b, *e) = s without leading and trailing White_Space scalars. */
void uni_trim(const char *s, size_t len, size_t *b, size_t *e);

#endif
