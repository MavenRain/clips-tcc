/* libav decode + swresample (ffmpeg-full) to the whisper.cpp input format. */
#ifndef CLIPS_AUDIO_H
#define CLIPS_AUDIO_H

#include <stddef.h>

enum { AUDIO_RATE = 16000 };

/* Decode the best audio stream of any libav-readable file to AUDIO_RATE Hz mono float32.
 * Timestamps are relative to the container start, with silence for audio delays and gaps.
 * Returns 0 on success; *pcm is malloc'd, *n is the sample count. media_sec (may be NULL) gets the
 * container duration, or the decoded length when libav has only a bitrate estimate or no duration. */
int audio_load(const char *path, float **pcm, size_t *n, double *media_sec);

#endif
