/* Stage 3 gate: SPEC 7.2 splitter cases, audio_load on the fixture, whisper words -> sentences. */
#include "audio.h"
#include "words.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavformat/avformat.h>
#include <libavutil/log.h>

#define VIDEO "fixtures/talk.mp4"
#define MODEL "models/ggml-base.en.bin"
#define TEXT "fixtures/talk.txt"

static int failures;

#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ---- 1. splitter units ---- */

#define W(t, s, e) { (char *)(t), (s), (e) }

/* Split ws and compare texts (NULL-terminated list) and, if given, start/end pairs. */
static void expect(const char *name, const cw_word *ws, size_t n, const char *const *texts, const double *times)
{
    cw_sentence *s = NULL;
    size_t k = 0, want = 0;
    while (texts[want]) want++;
    if (words_sentences(ws, n, &s, &k) != 0) {
        CHECK(0, "%s: splitter error", name);
        return;
    }
    CHECK(k == want, "%s: %zu sentences, want %zu", name, k, want);
    for (size_t i = 0; i < k && i < want; i++) {
        CHECK(strcmp(s[i].text, texts[i]) == 0, "%s: [%zu] \"%s\" want \"%s\"", name, i, s[i].text, texts[i]);
        if (times) CHECK(s[i].start == times[2 * i] && s[i].end == times[2 * i + 1],
                         "%s: [%zu] %.2f-%.2f want %.2f-%.2f", name, i, s[i].start, s[i].end, times[2 * i], times[2 * i + 1]);
    }
    printf("splitter %-22s %zu sentence(s)\n", name, k);
    sentences_free(s, k);
}

static void splitter_units(void)
{
    const cw_word two[] = { W(" Hello", 0.0, 0.5), W(" world.", 0.5, 1.0), W(" How", 1.2, 1.4), W(" are", 1.4, 1.6), W(" you?", 1.6, 2.0) };
    const char *two_t[] = { "Hello world.", "How are you?", NULL };
    const double two_s[] = { 0.0, 1.0, 1.2, 2.0 };
    expect("two sentences", two, 5, two_t, two_s);

    const cw_word lone[] = { W(".", 0.0, 0.1), W(" Next", 0.2, 0.4), W(" one", 0.4, 0.6), W(".", 0.6, 0.7) };
    const char *lone_t[] = { ". Next one.", NULL };
    expect("lone terminator", lone, 4, lone_t, (const double[]){ 0.0, 0.7 });

    const cw_word lead[] = { W(" ", 0.0, 0.1), W("\t ", 0.1, 0.2), W("", 0.2, 0.3), W(" Hi.", 0.5, 0.9) };
    const char *lead_t[] = { "Hi.", NULL };
    expect("leading whitespace", lead, 4, lead_t, (const double[]){ 0.5, 0.9 });

    const cw_word untimed[] = { W(" Lost", -1.0, -1.0), W(" words.", -1.0, -1.0), W(" Kept.", 3.0, 3.5) };
    const char *untimed_t[] = { "Kept.", NULL };
    expect("untimed run dropped", untimed, 3, untimed_t, (const double[]){ 3.0, 3.5 });

    /* "abcd" + 79 x " abc" trims to exactly 320 graphemes and closes; "abc" + 79 x " abc" (319) does not. */
    cw_word run[81];
    char big[400] = "abcd";
    for (int i = 0; i < 80; i++) run[i] = (cw_word)W(i ? " abc" : "abcd", i * 0.1, i * 0.1 + 0.1);
    for (int i = 1; i < 80; i++) strcat(big, " abc");
    run[80] = (cw_word)W(" tail", 9.0, 9.2);
    const char *run_t[] = { big, "tail", NULL };
    expect("run-on 320 closes", run, 81, run_t, (const double[]){ 0.0, 8.0, 9.0, 9.2 });
    run[0].text = "abc";
    memmove(big + 3, big + 4, strlen(big + 4) + 1);
    strcat(big, " tail");
    const char *run319_t[] = { big, NULL };
    expect("run-on 319 stays open", run, 81, run319_t, NULL);

    const cw_word ell[] = { W(" Wait", 0.0, 0.3), W("\xE2\x80\xA6", 0.3, 0.4), W(" then", 0.5, 0.7) };
    const char *ell_t[] = { "Wait\xE2\x80\xA6", "then", NULL };
    expect("U+2026 closes", ell, 3, ell_t, (const double[]){ 0.0, 0.4, 0.5, 0.7 });
}

/* ---- 2-3. fixture ---- */

/* Container duration and the best audio stream's own duration (seconds, -1 if unknown). */
static double container_seconds(const char *path, double *audio)
{
    AVFormatContext *fmt = NULL;
    double d = -1;
    *audio = -1;
    av_log_set_level(AV_LOG_ERROR);
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) return -1;
    if (avformat_find_stream_info(fmt, NULL) >= 0) {
        int si = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
        if (fmt->duration > 0) d = fmt->duration / (double)AV_TIME_BASE;
        if (si >= 0 && fmt->streams[si]->duration > 0)
            *audio = fmt->streams[si]->duration * av_q2d(fmt->streams[si]->time_base);
    }
    avformat_close_input(&fmt);
    return d;
}

static void print_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    char buf[4096];
    size_t got;
    if (!f) {
        printf("(no %s)\n", path);
        return;
    }
    printf("--- %s\n", path);
    while ((got = fread(buf, 1, sizeof buf, f)) > 0) fwrite(buf, 1, got, stdout);
    printf("\n---\n");
    fclose(f);
}

static void fixture(void)
{
    float *pcm = NULL;
    size_t n = 0, nw = 0, ns = 0;
    cw_word *w = NULL;
    cw_sentence *s = NULL;
    double adur = -1, dur = container_seconds(VIDEO, &adur);

    double media = -1;
    if (audio_load(VIDEO, &pcm, &n, &media) != 0) {
        CHECK(0, "audio_load failed");
        return;
    }
    /* talk.mp4: video track 94.0 s, AAC track 92.15 s; the decoded length matches the audio track. */
    double secs = n / (double)AUDIO_RATE;
    printf("audio: %zu samples = %.3f s, container %.3f s, audio stream %.3f s\n", n, secs, dur, adur);
    CHECK(adur > 0 && fabs(secs - adur) <= 0.5, "decoded %.3f s vs audio stream %.3f s", secs, adur);
    CHECK(dur > 0 && secs <= dur + 0.5, "decoded %.3f s exceeds container %.3f s", secs, dur);
    CHECK(fabs(media - dur) < 1e-9, "audio_load media %.3f s vs container %.3f s", media, dur);

    if (words_transcribe(MODEL, pcm, n, &w, &nw) != 0) {
        CHECK(0, "words_transcribe failed");
        free(pcm);
        return;
    }
    free(pcm);
    if (words_sentences(w, nw, &s, &ns) != 0) {
        CHECK(0, "words_sentences failed");
        words_free(w, nw);
        return;
    }
    printf("words %zu, sentences %zu\n", nw, ns);
    for (size_t i = 0; i < 3 && i < nw; i++) printf("word[%zu] \"%s\" %.2f-%.2f\n", i, w[i].text, w[i].start, w[i].end);
    for (size_t i = 0; i < 3 && i < ns; i++) printf("[%zu] %.2f-%.2f %s\n", i, s[i].start, s[i].end, s[i].text);
    CHECK(ns >= 3, "only %zu sentences", ns);
    for (size_t i = 0; i < ns; i++) {
        CHECK(s[i].start <= s[i].end, "sentence %zu start %.2f > end %.2f", i, s[i].start, s[i].end);
        if (i) CHECK(s[i - 1].start <= s[i].start, "sentence %zu start %.2f < previous %.2f", i, s[i].start, s[i - 1].start);
    }
    if (ns) CHECK(s[ns - 1].end <= dur + 0.5, "last end %.2f > duration %.2f + 0.5", s[ns - 1].end, dur);
    print_text(TEXT);
    sentences_free(s, ns);
    words_free(w, nw);
}

int main(void)
{
    splitter_units();
    fixture();
    printf(failures ? "check-words: %d FAILED\n" : "check-words: OK\n", failures);
    return failures ? 1 : 0;
}
