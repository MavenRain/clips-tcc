/* The 5 discourse features of the selector (SPEC 2.1-2.5). */
#ifndef CLIPS_FEATURES_H
#define CLIPS_FEATURES_H

/* out = [relative_position, hook_cue_count, payoff_cue_count, ends_with_question, digit_ratio]
 * for sentence index of n sentences. Returns 0 on success. */
int features_disc(const char *sentence, long index, long n, float out[5]);

/* Number of non-empty " "-separated tokens (SPEC 3 budget, SPEC 5.3 duration). */
long features_word_count(const char *s);

#endif
