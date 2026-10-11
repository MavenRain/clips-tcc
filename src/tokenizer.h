/* XLM-R SentencePiece Unigram over the RDTK container (SPEC section 1). */
#ifndef CLIPS_TOKENIZER_H
#define CLIPS_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct tok tok;

/* NULL on any reject rule of SPEC 1.1. */
tok *tok_load(const char *path);
tok *tok_from_bytes(const uint8_t *bytes, size_t n);

/* encode(text, max_len) per SPEC 1.2-1.4: NFKC, space squeeze, U+2581, Viterbi, bos/eos, truncation that keeps eos.
 * Writes at most max_len ids to out and returns the count; 0 when max_len <= 0; -1 on error. */
int tok_encode(const tok *t, const char *utf8, int max_len, int32_t *out);

void tok_free(tok *t);

#endif
