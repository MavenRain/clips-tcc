/* Feasibility probe: load the Clips Core ML package from plain C (TinyCC),
 * through the Objective-C runtime, and run both functions on a fixed batch.
 * Usage: coreml_probe [MODEL.mlmodelc] [computeUnits: 0 cpu, 1 cpu+gpu, 2 all, 3 cpu+ane] */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void *id;
typedef void *SEL;

enum { DT_F16 = 0x10010, DT_F32 = 0x10020, DT_F64 = 0x10040, DT_I32 = 0x20020, BATCH = 16 };

static id (*rt_class)(const char *);
static SEL (*rt_sel)(const char *);
static void *rt_msg;
static void *(*rt_pool_push)(void);
static void (*rt_pool_pop)(void *);

/* objc_msgSend must be called through its exact prototype on arm64. */
typedef id (*m_id)(id, SEL);
typedef id (*m_id_cstr)(id, SEL, const char *);
typedef id (*m_id_id)(id, SEL, id);
typedef id (*m_id_int)(id, SEL, int);
typedef id (*m_id_ul)(id, SEL, unsigned long);
typedef id (*m_id_err)(id, SEL, id, id *);
typedef id (*m_id_id_err)(id, SEL, id, id, id *);
typedef id (*m_id_long_err)(id, SEL, id, long, id *);
typedef id (*m_ptr_ul)(id, SEL, id *, unsigned long);
typedef id (*m_ptr_ptr_ul)(id, SEL, id *, id *, unsigned long);
typedef void (*v_id)(id, SEL, id);
typedef void (*v_long)(id, SEL, long);
typedef long (*l_void)(id, SEL);
typedef const char *(*s_void)(id, SEL);
typedef void *(*p_void)(id, SEL);

static id cls(const char *n) { return rt_class(n); }
static id msg(id o, const char *s) { return ((m_id)rt_msg)(o, rt_sel(s)); }
static long msg_long(id o, const char *s) { return ((l_void)rt_msg)(o, rt_sel(s)); }
static void *msg_ptr(id o, const char *s) { return ((p_void)rt_msg)(o, rt_sel(s)); }
static id ns(const char *s) { return ((m_id_cstr)rt_msg)(cls("NSString"), rt_sel("stringWithUTF8String:"), s); }
static id num(int v) { return ((m_id_int)rt_msg)(cls("NSNumber"), rt_sel("numberWithInt:"), v); }
static const char *cstr(id s) { return s ? ((s_void)rt_msg)(s, rt_sel("UTF8String")) : "(nil)"; }
static const char *desc(id o) { return o ? cstr(msg(o, "description")) : "(nil)"; }
static const char *errtext(id e) { return e ? cstr(msg(e, "localizedDescription")) : "(no NSError)"; }

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int rt_init(void) {
    void *objc = dlopen("/usr/lib/libobjc.A.dylib", RTLD_NOW);
    void *fdn = dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", RTLD_NOW);
    void *cml = dlopen("/System/Library/Frameworks/CoreML.framework/CoreML", RTLD_NOW);
    if (!objc || !fdn || !cml) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 0; }
    rt_class = dlsym(objc, "objc_getClass");
    rt_sel = dlsym(objc, "sel_registerName");
    rt_msg = dlsym(objc, "objc_msgSend");
    rt_pool_push = dlsym(objc, "objc_autoreleasePoolPush");
    rt_pool_pop = dlsym(objc, "objc_autoreleasePoolPop");
    return rt_class && rt_sel && rt_msg && rt_pool_push && rt_pool_pop;
}

static float half_to_float(uint16_t h) {
    union { uint32_t u; float f; } out;
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (exp == 0) { out.f = (float)man / 16777216.0f; out.u |= sign; return out.f; }
    out.u = sign | (exp == 31 ? 0x7f800000u | (man << 13) : ((exp + 112) << 23) | (man << 13));
    return out.f;
}

static long stride_at(id a, unsigned long i) {
    id s = ((m_id_ul)rt_msg)(msg(a, "strides"), rt_sel("objectAtIndex:"), i);
    return msg_long(s, "longValue");
}

static id load(const char *path, const char *fn, long units) {
    id cfg = msg(cls("MLModelConfiguration"), "new");
    ((v_id)rt_msg)(cfg, rt_sel("setFunctionName:"), ns(fn));
    ((v_long)rt_msg)(cfg, rt_sel("setComputeUnits:"), units);
    id url = ((m_id_id)rt_msg)(cls("NSURL"), rt_sel("fileURLWithPath:"), ns(path));
    id err = 0;
    double t0 = now_ms();
    id model = ((m_id_id_err)rt_msg)(cls("MLModel"), rt_sel("modelWithContentsOfURL:configuration:error:"), url, cfg, &err);
    if (!model) { fprintf(stderr, "[%s] load failed: %s\n", fn, errtext(err)); return 0; }
    id md = msg(model, "modelDescription");
    printf("[%s] load %.1f ms, units %ld\n", fn, now_ms() - t0, units);
    printf("[%s] inputs: %s\n", fn, desc(msg(md, "inputDescriptionsByName")));
    printf("[%s] outputs: %s\n", fn, desc(msg(md, "outputDescriptionsByName")));
    return model;
}

static id array2(long rows, long cols, long dtype) {
    id dims[2] = { num((int)rows), num((int)cols) };
    id shape = ((m_ptr_ul)rt_msg)(cls("NSArray"), rt_sel("arrayWithObjects:count:"), dims, 2UL);
    id err = 0;
    id a = ((m_id_long_err)rt_msg)(msg(cls("MLMultiArray"), "alloc"), rt_sel("initWithShape:dataType:error:"), shape, dtype, &err);
    if (!a) fprintf(stderr, "MLMultiArray: %s\n", errtext(err));
    return a;
}

/* Every row: [bos, 35378, 8999, 38, eos], rest zero-filled with mask 0. */
static void fill_tokens(id ids, id mask, long seq) {
    static const int32_t row[5] = { 0, 35378, 8999, 38, 2 };
    int32_t *pi = msg_ptr(ids, "dataPointer"), *pm = msg_ptr(mask, "dataPointer");
    long i0 = stride_at(ids, 0), i1 = stride_at(ids, 1), m0 = stride_at(mask, 0), m1 = stride_at(mask, 1);
    for (long r = 0; r < BATCH; r++)
        for (long c = 0; c < seq; c++) {
            pi[r * i0 + c * i1] = c < 5 ? row[c] : 0;
            pm[r * m0 + c * m1] = c < 5;
        }
}

static void fill_disc(id disc, const float *row) {
    float *p = msg_ptr(disc, "dataPointer");
    long s0 = stride_at(disc, 0), s1 = stride_at(disc, 1);
    for (long r = 0; r < BATCH; r++)
        for (long c = 0; c < 5; c++) p[r * s0 + c * s1] = row[c];
}

static float read_at(void *p, long dtype, long i) {
    if (dtype == DT_F16) return half_to_float(((uint16_t *)p)[i]);
    if (dtype == DT_F32) return ((float *)p)[i];
    if (dtype == DT_F64) return (float)((double *)p)[i];
    return (float)((int32_t *)p)[i];
}

static id predict(id model, const char *fn, id *keys, id *arrays, unsigned long n) {
    id vals[3];
    for (unsigned long i = 0; i < n; i++)
        vals[i] = ((m_id_id)rt_msg)(cls("MLFeatureValue"), rt_sel("featureValueWithMultiArray:"), arrays[i]);
    id dict = ((m_ptr_ptr_ul)rt_msg)(cls("NSDictionary"), rt_sel("dictionaryWithObjects:forKeys:count:"), vals, keys, n);
    id err = 0;
    id prov = ((m_id_err)rt_msg)(msg(cls("MLDictionaryFeatureProvider"), "alloc"), rt_sel("initWithDictionary:error:"), dict, &err);
    if (!prov) { fprintf(stderr, "[%s] provider: %s\n", fn, errtext(err)); return 0; }
    id out = 0;
    for (int pass = 0; pass < 2; pass++) {
        double t0 = now_ms();
        out = ((m_id_err)rt_msg)(model, rt_sel("predictionFromFeatures:error:"), prov, &err);
        if (!out) { fprintf(stderr, "[%s] predict failed: %s\n", fn, errtext(err)); return 0; }
        printf("[%s] predict pass %d: %.1f ms\n", fn, pass, now_ms() - t0);
    }
    return out;
}

static int dump(id out, const char *name) {
    id fv = ((m_id_id)rt_msg)(out, rt_sel("featureValueForName:"), ns(name));
    id a = fv ? msg(fv, "multiArrayValue") : 0;
    if (!a) { printf("  %s: MISSING\n", name); return 1; }
    long dt = msg_long(a, "dataType"), n = msg_long(a, "count"), s0 = stride_at(a, 0);
    void *p = msg_ptr(a, "dataPointer");
    printf("  %s: dtype 0x%lx shape %s strides %s count %ld\n   ", name, dt, desc(msg(a, "shape")), desc(msg(a, "strides")), n);
    float first = read_at(p, dt, 0);
    int bad = 0;
    for (long r = 0; r < BATCH && r < n; r++) {
        float v = read_at(p, dt, r * s0);
        bad |= !(v - v == 0.0f) || v != first;
        printf(" %.5f", v);
    }
    printf("\n   %s\n", bad ? "NOT all finite and equal" : "all finite and equal");
    return bad;
}

static int run_select(const char *path, long units) {
    static const float zero[5] = { 0, 0, 0, 0, 0 }, cue[5] = { 0.5f, 1, 0, 1, 0.1f };
    const float *discs[2] = { zero, cue };
    id model = load(path, "select", units);
    if (!model) return 1;
    id ids = array2(BATCH, 128, DT_I32), mask = array2(BATCH, 128, DT_I32), disc = array2(BATCH, 5, DT_F32);
    if (!ids || !mask || !disc) return 1;
    fill_tokens(ids, mask, 128);
    id keys[3] = { ns("ids"), ns("mask"), ns("disc") }, arrays[3] = { ids, mask, disc };
    int bad = 0;
    for (int v = 0; v < 2; v++) {
        fill_disc(disc, discs[v]);
        printf("[select] disc variant %d\n", v);
        id out = predict(model, "select", keys, arrays, 3);
        if (!out) return 1;
        bad |= dump(out, "saliency") | dump(out, "start_p") | dump(out, "end_p");
    }
    return bad;
}

static int run_score(const char *path, long units) {
    id model = load(path, "score", units);
    if (!model) return 1;
    id ids = array2(BATCH, 256, DT_I32), mask = array2(BATCH, 256, DT_I32);
    if (!ids || !mask) return 1;
    fill_tokens(ids, mask, 256);
    id keys[2] = { ns("ids"), ns("mask") }, arrays[2] = { ids, mask };
    id out = predict(model, "score", keys, arrays, 2);
    return out ? dump(out, "score") : 1;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models/clips.mlmodelc";
    long units = argc > 2 ? atol(argv[2]) : 2;
    if (!rt_init()) { fprintf(stderr, "objc runtime init failed\n"); return 1; }
    void *pool = rt_pool_push();
    int rc = run_select(path, units) | run_score(path, units);
    rt_pool_pop(pool);
    printf("VERDICT %s\n", rc ? "FAIL" : "OK");
    return rc;
}
