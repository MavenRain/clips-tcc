/* whisper.cpp words and the sentence splitter (SPEC section 7.1-7.2). */
#ifndef CLIPS_WORDS_H
#define CLIPS_WORDS_H

#include <stddef.h>

/* TimedWord (SPEC 7.1): text as emitted, leading space kept; times in seconds, negative when unplaced. */
typedef struct { char *text; double start, end; } cw_word;

/* Sentence (SPEC 7.2): id = index in the array; text trimmed. */
typedef struct { char *text; double start, end; } cw_sentence;

/* Transcribe AUDIO_RATE mono float32 with token timestamps. Returns 0 on success; *words malloc'd. */
int words_transcribe(const char *model_path, const float *pcm, size_t n, cw_word **words, size_t *count);

/* SPEC 7.2 splitter, runOnLimit 320. Returns 0 on success; *out malloc'd (may hold 0 sentences). */
int words_sentences(const cw_word *w, size_t n, cw_sentence **out, size_t *count);

void words_free(cw_word *w, size_t n);
void sentences_free(cw_sentence *s, size_t n);

#endif
