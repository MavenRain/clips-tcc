/* Timestamp regressions use tiny FFmpeg-generated videos, without speech models. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "audio.h"

#ifndef TEST_FFMPEG
#define TEST_FFMPEG "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
#endif
#define FF TEST_FFMPEG " -nostdin -hide_banner -loglevel error -y "

static int failed;

static int generate(const char *command)
{
    if (system(command) == 0) return 0;
    fprintf(stderr, "FAIL: generating audio timeline fixture\n");
    failed++;
    return -1;
}

static void onset(const char *path, double expected)
{
    float *pcm = NULL;
    size_t n = 0, first = 0;
    if (audio_load(path, &pcm, &n, NULL)) {
        fprintf(stderr, "FAIL: decoding %s\n", path);
        failed++;
        return;
    }
    while (first < n && fabsf(pcm[first]) < 0.01f) first++;
    double seconds = (double)first / AUDIO_RATE;
    double duration = (double)n / AUDIO_RATE;
    int ok = first < n && fabs(seconds - expected) < 0.15
             && fabs(duration - (expected + 1.0)) < 0.2;
    printf("audio onset: %s %.3f s, expected %.3f s, duration %.3f s\n",
           path, seconds, expected, duration);
    if (!ok) {
        fprintf(stderr, "FAIL: media-relative audio onset or duration\n");
        failed++;
    }
    free(pcm);
}

static void gap(const char *path)
{
    float *pcm = NULL;
    size_t n = 0, active = 0;
    if (audio_load(path, &pcm, &n, NULL)) {
        fprintf(stderr, "FAIL: decoding %s\n", path);
        failed++;
        return;
    }
    size_t lo = AUDIO_RATE * 3 / 5, hi = AUDIO_RATE * 13 / 10;
    for (size_t i = lo; i < hi && i < n; i++) active += fabsf(pcm[i]) > 0.01f;
    printf("audio gap: %.3f s, %zu active samples in the timestamp gap\n",
           (double)n / AUDIO_RATE, active);
    if (n < AUDIO_RATE * 12 / 5 || n > AUDIO_RATE * 27 / 10 || active != 0) {
        fprintf(stderr, "FAIL: timestamp gap must remain silent without shortening audio\n");
        failed++;
    }
    free(pcm);
}

/* Two 1 s tones back to back whose channel count or rate changes mid-stream; both must play in full. */
static void joined(const char *path, int decoded_duration)
{
    float *pcm = NULL;
    size_t n = 0, active = 0;
    double media = -1;
    if (audio_load(path, &pcm, &n, &media)) {
        fprintf(stderr, "FAIL: decoding %s\n", path);
        failed++;
        return;
    }
    size_t lo = AUDIO_RATE * 13 / 10, hi = AUDIO_RATE * 18 / 10;
    for (size_t i = lo; i < hi && i < n; i++) active += fabsf(pcm[i]) > 0.01f;
    printf("audio joined: %s %.3f s, %zu of %zu active samples in 1.3-1.8 s\n",
           path, (double)n / AUDIO_RATE, active, hi - lo);
    /* A 1 kHz tone at 16 kHz puts 2 of every 16 samples near a zero crossing. */
    if (n < AUDIO_RATE * 19 / 10 || n > AUDIO_RATE * 9 / 4 || active < (hi - lo) * 3 / 4) {
        fprintf(stderr, "FAIL: a mid-stream format change must not drop, speed up or crash audio\n");
        failed++;
    }
    if (decoded_duration && fabs(media - (double)n / AUDIO_RATE) > 1.0 / AUDIO_RATE) {
        fprintf(stderr, "FAIL: raw audio duration %.6f must use the decoded timeline %.6f\n",
                media, (double)n / AUDIO_RATE);
        failed++;
    }
    free(pcm);
}

/* The first sound and the decoded length must stay within 20 ms of the container's own timeline. */
static void timeline(const char *path, double start, double length)
{
    float *pcm = NULL;
    size_t n = 0, first = 0;
    if (audio_load(path, &pcm, &n, NULL)) {
        fprintf(stderr, "FAIL: decoding %s\n", path);
        failed++;
        return;
    }
    while (first < n && fabsf(pcm[first]) < 0.01f) first++;
    double seconds = (double)first / AUDIO_RATE;
    double duration = (double)n / AUDIO_RATE;
    printf("audio timeline: %s onset %.3f s, expected %.3f s, length %.3f s, expected %.3f s\n",
           path, seconds, start, duration, length);
    if (first == n || fabs(seconds - start) > 0.02 || fabs(duration - length) > 0.02) {
        fprintf(stderr, "FAIL: libav timestamp fill-in must keep the onset and length\n");
        failed++;
    }
    free(pcm);
}

#define TONE(rate, ch, out) FF "-f lavfi -i sine=frequency=1000:duration=1:sample_rate=" rate " -ac " ch " -c:a aac " out

int main(void)
{
    if (!generate(FF "-f lavfi -i color=c=black:s=16x16:r=1:d=6 "
                  "-itsoffset 5 -f lavfi -i sine=frequency=1000:duration=1:sample_rate=16000 "
                  "-map 0:v -map 1:a -c:v mpeg4 -c:a aac build/test-audio-delayed.mp4"))
        onset("build/test-audio-delayed.mp4", 5.0);
    if (!generate(FF "-itsoffset 5 -f lavfi -i color=c=black:s=16x16:r=1:d=1 "
                  "-itsoffset 5 -f lavfi -i sine=frequency=1000:duration=1:sample_rate=16000 "
                  "-map 0:v -map 1:a -c:v mpeg4 -c:a aac build/test-audio-origin.mp4"))
        onset("build/test-audio-origin.mp4", 0.064);
    if (!generate(FF "-f lavfi -i color=c=black:s=16x16:r=1:d=3 "
                  "-f lavfi -i sine=frequency=1000:duration=2.5:sample_rate=16000 "
                  "-map 0:v -map 1:a -af \"aselect='not(between(t,0.45,1.45))'\" "
                  "-c:v mpeg4 -c:a aac build/test-audio-gap.mp4"))
        gap("build/test-audio-gap.mp4");
    if (!generate(FF "-f lavfi -i color=c=black:s=16x16:r=1:d=3 "
                  "-f lavfi -i sine=frequency=1000:duration=2.5:sample_rate=44100 "
                  "-map 0:v -map 1:a -af \"aselect='not(between(t,0.45,1.45))'\" "
                  "-c:v mpeg4 -c:a aac build/test-audio-resample-gap.mp4"))
        gap("build/test-audio-resample-gap.mp4");
    /* MPEG-TS needs the fill-in for its start time, MP3 for its priming trim. */
    /* The lengths 2.112 and 3.048 include the encoder priming of ffmpeg 9.0.2. */
    if (!generate(FF "-f lavfi -i sine=frequency=1000:duration=2:sample_rate=16000 "
                  "-c:a aac -output_ts_offset 10 -f mpegts build/test-audio-ts-offset.ts"))
        timeline("build/test-audio-ts-offset.ts", 0.064, 2.112);
    if (!generate(FF "-f lavfi -i color=c=black:s=16x16:r=1:d=3 "
                  "-itsoffset 1 -f lavfi -i sine=frequency=1000:duration=2:sample_rate=16000 "
                  "-map 0:v -map 1:a -c:v mpeg4 -c:a aac -f mpegts build/test-audio-ts-delayed.ts"))
        timeline("build/test-audio-ts-delayed.ts", 1.0, 3.048);
    if (!generate(FF "-f lavfi -i sine=frequency=1000:duration=2:sample_rate=16000 "
                  "-c:a libmp3lame build/test-audio.mp3"))
        timeline("build/test-audio.mp3", 0.0, 2.0);
    /* AC3 exercises a layout copy whose destination must be initialized before libav frees it. */
    if (!generate(FF "-f lavfi -i sine=frequency=1000:duration=2:sample_rate=32000 "
                  "-c:a ac3 build/test-audio.ac3"))
        timeline("build/test-audio.ac3", 0.0, 2.016);
    /* A rate change mid-stream rebuilds the resampler, so the AC3 layout copy runs a second time. */
    if (!generate(FF "-f lavfi -i sine=frequency=1000:duration=1:sample_rate=32000 "
                  "-c:a ac3 build/test-audio-ac3-low.ac3")
        && !generate(FF "-f lavfi -i sine=frequency=1000:duration=1:sample_rate=48000 "
                     "-c:a ac3 build/test-audio-ac3-high.ac3")
        && !generate("cat build/test-audio-ac3-low.ac3 build/test-audio-ac3-high.ac3 "
                     "> build/test-audio-ac3-rate-up.ac3")) {
        joined("build/test-audio-ac3-rate-up.ac3", 1);
        timeline("build/test-audio-ac3-rate-up.ac3", 0.0, 2.032);
    }
    /* The second TS segment starts where the first ends. Its PES packets contain several AAC
     * frames, so parser fill-in must not use 22050 Hz for the new 44100 Hz frames. */
    if (!generate(FF "-f lavfi -i sine=frequency=1000:duration=1:sample_rate=22050 "
                  "-c:a aac -f mpegts build/test-audio-ts-low.ts")
        && !generate(FF "-f lavfi -i sine=frequency=1000:duration=1:sample_rate=44100 "
                     "-c:a aac -output_ts_offset 1.091337 -f mpegts build/test-audio-ts-high.ts")
        && !generate("cat build/test-audio-ts-low.ts build/test-audio-ts-high.ts > build/test-audio-ts-rate-up.ts"))
        joined("build/test-audio-ts-rate-up.ts", 0);
    /* Raw ADTS files concatenate as-is, so the decoder sees the new layout or rate mid-stream. */
    if (!generate(TONE("44100", "2", "build/test-audio-stereo.aac"))
        && !generate(TONE("44100", "1", "build/test-audio-mono.aac"))
        && !generate(TONE("22050", "2", "build/test-audio-low.aac"))
        && !generate("cat build/test-audio-stereo.aac build/test-audio-mono.aac > build/test-audio-layout.aac")
        && !generate("cat build/test-audio-stereo.aac build/test-audio-low.aac > build/test-audio-rate.aac")) {
        joined("build/test-audio-layout.aac", 1);
        joined("build/test-audio-rate.aac", 1);
        if (!generate("cat build/test-audio-low.aac build/test-audio-stereo.aac > build/test-audio-rate-up.aac"))
            joined("build/test-audio-rate-up.aac", 1);
    }
    printf("test_audio: 14 scenarios, %d failed\n", failed);
    return failed ? 1 : 0;
}
