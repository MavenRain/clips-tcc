/* libav decode + swresample to AUDIO_RATE mono float32 (the whisper.cpp input format). */
#include "audio.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/log.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>

/* Growable float buffer; capacity doubles on demand. */
typedef struct { float *p; size_t n, cap; } fbuf;

/* Decode state shared by the helpers. */
typedef struct {
    AVCodecContext *dec;
    AVFrame *fr;
    SwrContext *swr;
    AVChannelLayout in_layout; /* input side of swr: a frame that differs rebuilds it */
    int in_rate, in_format;
    AVRational time_base;
    int64_t origin_samples;
    fbuf out;
} dstate;

static int fbuf_reserve(fbuf *b, size_t extra)
{
    size_t need = b->n + extra, cap = b->cap ? b->cap : AUDIO_RATE;
    if (need <= b->cap) return 0;
    while (cap < need) cap *= 2;
    float *p = realloc(b->p, cap * sizeof *p);
    if (!p) return -1;
    b->p = p;
    b->cap = cap;
    return 0;
}

/* Build the resampler for frame f (its layout, format and rate are authoritative). */
static int swr_open(dstate *d, const AVFrame *f)
{
    AVChannelLayout mono = {0}, in = {0};
    av_channel_layout_uninit(&d->in_layout);
    if (av_channel_layout_copy(&d->in_layout, &f->ch_layout) < 0) return -1;
    d->in_rate = f->sample_rate;
    d->in_format = f->format;
    av_channel_layout_default(&mono, 1);
    if (f->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) av_channel_layout_default(&in, f->ch_layout.nb_channels);
    else if (av_channel_layout_copy(&in, &f->ch_layout) < 0) return -1;
    int r = swr_alloc_set_opts2(&d->swr, &mono, AV_SAMPLE_FMT_FLT, AUDIO_RATE,
                                &in, (enum AVSampleFormat)f->format, f->sample_rate, 0, NULL);
    av_channel_layout_uninit(&in);
    if (r < 0 || !d->swr) return -1;
    return swr_init(d->swr) < 0 ? -1 : 0;
}

/* Convert one frame into d->out; f NULL drains the resampler. Returns samples written or -1. */
static int resample(dstate *d, const AVFrame *f)
{
    int in = f ? f->nb_samples : 0;
    int room = swr_get_out_samples(d->swr, in);
    if (room < 0) return -1;
    room += 64;
    if (fbuf_reserve(&d->out, (size_t)room) < 0) return -1;
    uint8_t *dst = (uint8_t *)(d->out.p + d->out.n);
    int got = swr_convert(d->swr, &dst, room, f ? (const uint8_t **)f->extended_data : NULL, in);
    if (got < 0) return -1;
    d->out.n += (size_t)got;
    return got;
}

/* True when f's layout, format or rate differs from the input the resampler was built for. */
static int input_changed(const dstate *d, const AVFrame *f)
{
    return f->sample_rate != d->in_rate || f->format != d->in_format
        || av_channel_layout_compare(&f->ch_layout, &d->in_layout) != 0;
}

/* Flush the resampler, pad d->out with silence up to sample at (if later), and rebuild it for f. */
static int swr_restart(dstate *d, const AVFrame *f, int64_t at)
{
    int flushed;
    while ((flushed = resample(d, NULL)) > 0) {}
    if (flushed < 0) return -1;
    if (at > (int64_t)d->out.n) {
        size_t gap = (size_t)(at - (int64_t)d->out.n);
        if (fbuf_reserve(&d->out, gap) < 0) return -1;
        memset(d->out.p + d->out.n, 0, gap * sizeof *d->out.p);
        d->out.n += gap;
    }
    swr_free(&d->swr);
    return swr_open(d, f);
}

/* Receive every ready frame and resample it. Returns 0 at EAGAIN or EOF, -1 on error. */
static int drain(dstate *d)
{
    int r;
    while ((r = avcodec_receive_frame(d->dec, d->fr)) >= 0) {
        int e = d->swr ? 0 : swr_open(d, d->fr);
        int64_t at = -1;
        if (e == 0 && d->fr->best_effort_timestamp != AV_NOPTS_VALUE) {
            int64_t t = av_rescale_q(d->fr->best_effort_timestamp, d->time_base,
                                     (AVRational){1, AUDIO_RATE}) - d->origin_samples;
            /* Allow two samples for timestamp and resampler rounding. */
            if (t > 0 && t > (int64_t)d->out.n + swr_get_delay(d->swr, AUDIO_RATE) + 2) at = t;
        }
        /* A gap pads with silence; a new layout, format or rate (ADTS, TS) needs a new resampler. */
        if (e == 0 && (at >= 0 || input_changed(d, d->fr))) e = swr_restart(d, d->fr, at);
        if (e == 0) e = resample(d, d->fr);
        av_frame_unref(d->fr);
        if (e < 0) return -1;
    }
    return (r == AVERROR(EAGAIN) || r == AVERROR_EOF) ? 0 : -1;
}

/* Feed one packet; invalid data in a single packet is skipped as ffmpeg does. */
static int feed(dstate *d, const AVPacket *pkt)
{
    int e = avcodec_send_packet(d->dec, pkt);
    if (e < 0 && e != AVERROR_INVALIDDATA) return -1;
    return drain(d);
}

static int fail(const char *path, const char *what)
{
    fprintf(stderr, "audio: %s: %s\n", path, what);
    return -1;
}

/* Open the best audio stream's decoder. Returns the stream index or -1. */
static int open_decoder(AVFormatContext *fmt, dstate *d)
{
    const AVCodec *codec = NULL;
    int si = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (si < 0 || !codec) return -1;
    d->dec = avcodec_alloc_context3(codec);
    if (!d->dec) return -1;
    if (avcodec_parameters_to_context(d->dec, fmt->streams[si]->codecpar) < 0) return -1;
    return avcodec_open2(d->dec, codec, NULL) < 0 ? -1 : si;
}

/* Read, decode and resample every packet of stream si, then flush decoder and resampler. */
static const char *decode_all(AVFormatContext *fmt, int si, dstate *d)
{
    AVPacket *pkt = av_packet_alloc();
    if (!pkt) return "out of memory";
    const char *err = NULL;
    int r;
    while (!err && (r = av_read_frame(fmt, pkt)) >= 0) {
        if (pkt->stream_index == si && feed(d, pkt) < 0) err = "decode failed";
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    if (err) return err;
    if (r != AVERROR_EOF) return "read failed";
    if (feed(d, NULL) < 0) return "decoder flush failed";
    if (!d->swr) return "no audio decoded";
    while ((r = resample(d, NULL)) > 0) {}
    return r < 0 ? "resampler flush failed" : NULL;
}

/* Probe the container timeline with fill-in, then discard synthesized parser timestamps for TS.
 * AAC inside one PES can change rate; fill-in still uses the stream's original rate between PES
 * timestamps. A fresh demuxer retains stored timestamps without keeping the probe's filled queue. */
static int stream_info(const char *path, AVFormatContext **fmt, double *media, int64_t *start)
{
    if (avformat_find_stream_info(*fmt, NULL) < 0) return -1;
    *start = (*fmt)->start_time;
    if ((*fmt)->duration > 0 && (*fmt)->duration_estimation_method != AVFMT_DURATION_FROM_BITRATE)
        *media = (double)(*fmt)->duration / AV_TIME_BASE;
    if (!strcmp((*fmt)->iformat->name, "mpegts")) {
        avformat_close_input(fmt);
        if (avformat_open_input(fmt, path, NULL, NULL) < 0) return -1;
        (*fmt)->flags |= AVFMT_FLAG_NOFILLIN;
        if (avformat_find_stream_info(*fmt, NULL) < 0) return -1;
    }
    return 0;
}

int audio_load(const char *path, float **pcm, size_t *n, double *media_sec)
{
    AVFormatContext *fmt = NULL;
    dstate d = {0};
    const char *err = NULL;
    double media = -1;
    int64_t start = AV_NOPTS_VALUE;
    int si;

    *pcm = NULL;
    *n = 0;
    if (media_sec) *media_sec = 0;
    av_log_set_level(AV_LOG_ERROR);
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) return fail(path, "cannot open");
    /* Raw AAC (ADTS, LOAS) stores no timestamps, and libav infers them with the first sample rate
     * after a rate change, so drain() would insert silence that is absent from the input. Other
     * demuxers keep the fill-in: MPEG-TS needs it for start_time, MP3 for its priming trim. */
    if (!strcmp(fmt->iformat->name, "aac") || !strcmp(fmt->iformat->name, "loas"))
        fmt->flags |= AVFMT_FLAG_NOFILLIN;
    if (stream_info(path, &fmt, &media, &start) < 0) err = "no stream info";
    else if ((si = open_decoder(fmt, &d)) < 0) err = "no decodable audio stream";
    else if (!(d.fr = av_frame_alloc())) err = "out of memory";
    else {
        /* The CLI's material length: the whole container, not the audio track. */
        d.time_base = fmt->streams[si]->time_base;
        d.origin_samples = start == AV_NOPTS_VALUE ? 0
            : av_rescale_q(start, (AVRational){1, AV_TIME_BASE},
                           (AVRational){1, AUDIO_RATE});
        err = decode_all(fmt, si, &d);
    }

    av_frame_free(&d.fr);
    swr_free(&d.swr);
    av_channel_layout_uninit(&d.in_layout);
    avcodec_free_context(&d.dec);
    avformat_close_input(&fmt);
    if (err) {
        free(d.out.p);
        return fail(path, err);
    }
    *pcm = d.out.p;
    *n = d.out.n;
    if (media_sec) *media_sec = media > 0 ? media : (double)d.out.n / AUDIO_RATE;
    return 0;
}
