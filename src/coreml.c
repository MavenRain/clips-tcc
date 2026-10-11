/* Core ML through the Objective-C runtime (SPEC section 6, probe/PROBE.md).
 * libobjc, Foundation and CoreML are dlopen'ed; every objc_msgSend goes through
 * the exact prototype of its call site (required on arm64). Manual retain/release:
 * new/alloc results are owned, the rest are autoreleased and retained only when
 * the model keeps them. Input dtype conversion mirrors CoreMLSession.swift:237-313. */
#include "coreml.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *id;
typedef void *SEL;

/* MLMultiArrayDataType values from the Core ML SDK headers (SPEC section 6). */
enum { DT_F16 = 0x10010, DT_F32 = 0x10020, DT_F64 = 0x10040, DT_I32 = 0x20020 };
enum { IN_IDS, IN_MASK, IN_DISC, MAX_IN = 3, MAX_RANK = 8 };

struct input {
    int kind;
    long dtype, s0, s1; /* declared dtype; strides of the [CML_BATCH, cols] array */
    id name, array;     /* retained NSString, owned MLMultiArray */
};

struct cml_model {
    char fn[32];
    id model; /* retained MLModel */
    long width;
    int n_in;
    struct input in[MAX_IN];
};

static id (*rt_class)(const char *);
static SEL (*rt_sel)(const char *);
static void *rt_msg;
static void *(*rt_pool_push)(void);
static void (*rt_pool_pop)(void *);

typedef id (*m_id)(id, SEL);
typedef id (*m_id_cstr)(id, SEL, const char *);
typedef id (*m_id_id)(id, SEL, id);
typedef id (*m_id_long)(id, SEL, long);
typedef id (*m_id_ul)(id, SEL, unsigned long);
typedef id (*m_id_err)(id, SEL, id, id *);
typedef id (*m_id_id_err)(id, SEL, id, id, id *);
typedef id (*m_id_long_err)(id, SEL, id, long, id *);
typedef id (*m_ptr_ul)(id, SEL, id *, unsigned long);
typedef id (*m_ptr_ptr_ul)(id, SEL, id *, id *, unsigned long);
typedef void (*v_void)(id, SEL);
typedef void (*v_id)(id, SEL, id);
typedef void (*v_long)(id, SEL, long);
typedef long (*l_void)(id, SEL);
typedef const char *(*s_void)(id, SEL);
typedef void *(*p_void)(id, SEL);

static id cls(const char *n) { return rt_class(n); }
static id msg(id o, const char *s) { return ((m_id)rt_msg)(o, rt_sel(s)); }
static long msg_long(id o, const char *s) { return ((l_void)rt_msg)(o, rt_sel(s)); }
static void *msg_ptr(id o, const char *s) { return ((p_void)rt_msg)(o, rt_sel(s)); }
static id at(id a, long i) { return ((m_id_ul)rt_msg)(a, rt_sel("objectAtIndex:"), (unsigned long)i); }
static id ns(const char *s) { return ((m_id_cstr)rt_msg)(cls("NSString"), rt_sel("stringWithUTF8String:"), s); }
static id num(long v) { return ((m_id_long)rt_msg)(cls("NSNumber"), rt_sel("numberWithLong:"), v); }
static id retain(id o) { return o ? msg(o, "retain") : 0; }
static void release(id o) { if (o) ((v_void)rt_msg)(o, rt_sel("release")); }

static const char *cstr(id s) {
    const char *c = s ? ((s_void)rt_msg)(s, rt_sel("UTF8String")) : 0;
    return c ? c : "(nil)";
}

static const char *errtext(id e) { return e ? cstr(msg(e, "localizedDescription")) : "(no NSError)"; }

int cml_init(void) {
    if (rt_msg) return 0;
    /* The handles stay open for the life of the process: the runtime is never unloaded. */
    void *objc = dlopen("/usr/lib/libobjc.A.dylib", RTLD_NOW);
    void *fdn = dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", RTLD_NOW);
    void *cml = dlopen("/System/Library/Frameworks/CoreML.framework/CoreML", RTLD_NOW);
    if (!objc || !fdn || !cml) { fprintf(stderr, "cml_init: dlopen: %s\n", dlerror()); return -1; }
    rt_class = dlsym(objc, "objc_getClass");
    rt_sel = dlsym(objc, "sel_registerName");
    rt_pool_push = dlsym(objc, "objc_autoreleasePoolPush");
    rt_pool_pop = dlsym(objc, "objc_autoreleasePoolPop");
    void *send = dlsym(objc, "objc_msgSend");
    if (!rt_class || !rt_sel || !rt_pool_push || !rt_pool_pop || !send) {
        fprintf(stderr, "cml_init: objc runtime symbols missing\n");
        return -1;
    }
    rt_msg = send;
    return 0;
}

static float half_to_float(uint16_t h) {
    union { uint32_t u; float f; } out;
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (exp == 0) { out.f = (float)man / 16777216.0f; out.u |= sign; return out.f; }
    out.u = sign | (exp == 31 ? 0x7f800000u | (man << 13) : ((exp + 112) << 23) | (man << 13));
    return out.f;
}

/* Round to nearest even, as vImageConvert_PlanarFtoPlanar16F does. */
static uint16_t float_to_half(float f) {
    union { float f; uint32_t u; } in;
    in.f = f;
    uint32_t sign = (in.u >> 16) & 0x8000, exp = (in.u >> 23) & 0xff, man = in.u & 0x7fffff;
    int e = (int)exp - 112; /* rebias 127 -> 15 */
    if (exp == 255) return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e < -10) return (uint16_t)sign;
    if (e <= 0) { /* subnormal: shift the 24-bit significand into 10 bits */
        uint32_t full = man | 0x800000, shift = (uint32_t)(14 - e);
        uint32_t half = full >> shift, rem = full & ((1u << shift) - 1), mid = 1u << (shift - 1);
        half += rem > mid || (rem == mid && (half & 1));
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)e << 10) | (man >> 13), rem = man & 0x1fff;
    half += rem > 0x1000 || (rem == 0x1000 && (half & 1)); /* a carry into exp 31 gives inf */
    return (uint16_t)(sign | half);
}

static int dtype_ok(long dt) { return dt == DT_F16 || dt == DT_F32 || dt == DT_F64 || dt == DT_I32; }

/* Store v at element i of a dtype dt buffer; dt is checked with dtype_ok first. */
static void put(void *p, long dt, long i, double v) {
    switch (dt) {
    case DT_I32: ((int32_t *)p)[i] = (int32_t)v; break;
    case DT_F16: ((uint16_t *)p)[i] = float_to_half((float)v); break;
    case DT_F32: ((float *)p)[i] = (float)v; break;
    case DT_F64: ((double *)p)[i] = v; break;
    default: break;
    }
}

/* Element i of a dtype dt buffer, widened to float. */
static float get(const void *p, long dt, long i) {
    switch (dt) {
    case DT_I32: return (float)((const int32_t *)p)[i];
    case DT_F16: return half_to_float(((const uint16_t *)p)[i]);
    case DT_F32: return ((const float *)p)[i];
    case DT_F64: return (float)((const double *)p)[i];
    default: return 0.0f;
    }
}

/* Copy an NSArray of NSNumber into v; the count, or -1 when it exceeds max. */
static int longs(id arr, long *v, int max) {
    long n = arr ? msg_long(arr, "count") : 0;
    if (n > max) return -1;
    for (long i = 0; i < n; i++) v[i] = msg_long(at(arr, i), "longValue");
    return (int)n;
}

/* Buffer offset of the i-th element in row-major logical order. */
static long offset_of(long i, int rank, const long *shape, const long *strides) {
    long off = 0;
    for (int d = rank - 1; d >= 0; d--) {
        long n = shape[d] > 0 ? shape[d] : 1;
        off += (i % n) * strides[d];
        i /= n;
    }
    return off;
}

static id array2(long rows, long cols, long dtype) {
    id dims[2] = { num(rows), num(cols) };
    id shape = ((m_ptr_ul)rt_msg)(cls("NSArray"), rt_sel("arrayWithObjects:count:"), dims, 2UL);
    id err = 0;
    id a = ((m_id_long_err)rt_msg)(msg(cls("MLMultiArray"), "alloc"), rt_sel("initWithShape:dataType:error:"), shape, dtype, &err);
    if (!a) fprintf(stderr, "MLMultiArray [%ld, %ld] dtype 0x%lx: %s\n", rows, cols, dtype, errtext(err));
    return a;
}

/* The multiArrayConstraint of input name, or nil. */
static id constraint(id inputs, id name) {
    id fd = ((m_id_id)rt_msg)(inputs, rt_sel("objectForKey:"), name);
    return fd ? msg(fd, "multiArrayConstraint") : 0;
}

static int open_model(cml_model *m, const char *path, long units) {
    id cfg = msg(cls("MLModelConfiguration"), "new");
    if (!cfg) { fprintf(stderr, "[%s] MLModelConfiguration unavailable\n", m->fn); return -1; }
    ((v_id)rt_msg)(cfg, rt_sel("setFunctionName:"), ns(m->fn));
    ((v_long)rt_msg)(cfg, rt_sel("setComputeUnits:"), units);
    id url = ((m_id_id)rt_msg)(cls("NSURL"), rt_sel("fileURLWithPath:"), ns(path));
    id err = 0;
    id model = ((m_id_id_err)rt_msg)(cls("MLModel"), rt_sel("modelWithContentsOfURL:configuration:error:"), url, cfg, &err);
    release(cfg);
    if (!model) { fprintf(stderr, "[%s] load %s failed: %s\n", m->fn, path, errtext(err)); return -1; }
    m->model = retain(model);
    return 0;
}

/* Keep one declared input: its dtype and a reusable [CML_BATCH, cols] array of that dtype. */
static int add_input(cml_model *m, id inputs, id name) {
    const char *s = cstr(name);
    int kind = !strcmp(s, "ids") ? IN_IDS : !strcmp(s, "mask") ? IN_MASK : !strcmp(s, "disc") ? IN_DISC : -1;
    id con = constraint(inputs, name);
    if (kind < 0 || !con) { fprintf(stderr, "[%s] input %s: not ids, mask or disc as a multi-array\n", m->fn, s); return -1; }
    struct input *in = &m->in[m->n_in++];
    long st[MAX_RANK];
    in->kind = kind;
    in->name = retain(name);
    in->dtype = msg_long(con, "dataType");
    if (!dtype_ok(in->dtype)) { fprintf(stderr, "[%s] input %s: dtype 0x%lx not supported\n", m->fn, s, in->dtype); return -1; }
    in->array = array2(CML_BATCH, kind == IN_DISC ? CML_DISC : m->width, in->dtype);
    if (!in->array) return -1;
    if (longs(msg(in->array, "strides"), st, MAX_RANK) != 2) { fprintf(stderr, "[%s] input %s: no strides\n", m->fn, s); return -1; }
    in->s0 = st[0];
    in->s1 = st[1];
    return 0;
}

/* Read the input description once: the ids width, then every declared input. */
static int describe(cml_model *m) {
    id inputs = msg(msg(m->model, "modelDescription"), "inputDescriptionsByName");
    id ids = constraint(inputs, ns("ids"));
    long shape[MAX_RANK];
    if (!ids || longs(msg(ids, "shape"), shape, MAX_RANK) < 2 || shape[1] <= 0) {
        fprintf(stderr, "[%s] input ids: no [batch, width] shape in the model description\n", m->fn);
        return -1;
    }
    m->width = shape[1];
    id keys = msg(inputs, "allKeys");
    long n = msg_long(keys, "count");
    if (n > MAX_IN) { fprintf(stderr, "[%s] %ld inputs declared, at most %d supported\n", m->fn, n, MAX_IN); return -1; }
    for (long i = 0; i < n; i++)
        if (add_input(m, inputs, at(keys, i))) return -1;
    return 0;
}

cml_model *cml_load(const char *path, const char *function, long units) {
    if (!path || !function) { fprintf(stderr, "cml_load: NULL path or function\n"); return NULL; }
    if (cml_init()) return NULL;
    cml_model *m = calloc(1, sizeof *m);
    if (!m) { fprintf(stderr, "cml_load: out of memory\n"); return NULL; }
    snprintf(m->fn, sizeof m->fn, "%s", function);
    void *pool = rt_pool_push();
    int rc = open_model(m, path, units) || describe(m);
    rt_pool_pop(pool);
    if (rc) { cml_free(m); return NULL; }
    return m;
}

long cml_width(const cml_model *m, long fallback) { return m && m->width > 0 ? m->width : fallback; }

/* Convert the host row-major buffer (int32 si or float32 sf) into the input array, by strides. */
static int fill(const cml_model *m, const struct input *in, const int32_t *si, const float *sf) {
    long cols = in->kind == IN_DISC ? CML_DISC : m->width;
    void *p = msg_ptr(in->array, "dataPointer");
    if (!p) { fprintf(stderr, "[%s] input %s: no data pointer\n", m->fn, cstr(in->name)); return -1; }
    for (long r = 0; r < CML_BATCH; r++)
        for (long c = 0; c < cols; c++) {
            long h = r * cols + c;
            put(p, in->dtype, r * in->s0 + c * in->s1, si ? (double)si[h] : (double)sf[h]);
        }
    return 0;
}

/* The first CML_BATCH logical elements of output name, any dtype, by strides. */
static int read_output(const cml_model *m, id out, const char *name, float *dst) {
    id fv = ((m_id_id)rt_msg)(out, rt_sel("featureValueForName:"), ns(name));
    id a = fv ? msg(fv, "multiArrayValue") : 0;
    if (!a) { fprintf(stderr, "[%s] output %s: missing or not a multi-array\n", m->fn, name); return -1; }
    long dt = msg_long(a, "dataType"), count = msg_long(a, "count"), shape[MAX_RANK], strides[MAX_RANK];
    int rank = longs(msg(a, "shape"), shape, MAX_RANK);
    const void *p = msg_ptr(a, "dataPointer");
    if (!dtype_ok(dt) || rank < 1 || longs(msg(a, "strides"), strides, MAX_RANK) != rank || !p || count < CML_BATCH) {
        fprintf(stderr, "[%s] output %s: dtype 0x%lx rank %d count %ld not readable\n", m->fn, name, dt, rank, count);
        return -1;
    }
    for (long r = 0; r < CML_BATCH; r++) dst[r] = get(p, dt, offset_of(r, rank, shape, strides));
    return 0;
}

static int run(cml_model *m, const int32_t *ids, const int32_t *mask, const float *disc,
               const char *const *names, int n_out, float *const *outs) {
    id vals[MAX_IN], keys[MAX_IN];
    for (int i = 0; i < m->n_in; i++) {
        const struct input *in = &m->in[i];
        const int32_t *si = in->kind == IN_IDS ? ids : in->kind == IN_MASK ? mask : 0;
        const float *sf = in->kind == IN_DISC ? disc : 0;
        if (!si && !sf) { fprintf(stderr, "[%s] input %s: no host buffer\n", m->fn, cstr(in->name)); return -1; }
        if (fill(m, in, si, sf)) return -1;
        vals[i] = ((m_id_id)rt_msg)(cls("MLFeatureValue"), rt_sel("featureValueWithMultiArray:"), in->array);
        keys[i] = in->name;
        if (!vals[i]) { fprintf(stderr, "[%s] input %s: no feature value\n", m->fn, cstr(in->name)); return -1; }
    }
    id dict = ((m_ptr_ptr_ul)rt_msg)(cls("NSDictionary"), rt_sel("dictionaryWithObjects:forKeys:count:"), vals, keys, (unsigned long)m->n_in);
    if (!dict) { fprintf(stderr, "[%s] input dictionary: nil\n", m->fn); return -1; }
    id err = 0;
    id prov = ((m_id_err)rt_msg)(msg(cls("MLDictionaryFeatureProvider"), "alloc"), rt_sel("initWithDictionary:error:"), dict, &err);
    if (!prov) { fprintf(stderr, "[%s] provider: %s\n", m->fn, errtext(err)); return -1; }
    id out = ((m_id_err)rt_msg)(m->model, rt_sel("predictionFromFeatures:error:"), prov, &err);
    int rc = out ? 0 : -1;
    if (!out) fprintf(stderr, "[%s] predict failed: %s\n", m->fn, errtext(err));
    for (int k = 0; k < n_out && !rc; k++) rc = read_output(m, out, names[k], outs[k]);
    release(prov);
    return rc;
}

int cml_predict(cml_model *m, const int32_t *ids, const int32_t *mask, const float *disc,
                const char *const *names, int n_out, float *const *outs) {
    if (!m || !rt_msg) { fprintf(stderr, "cml_predict: no model\n"); return -1; }
    if (n_out < 0 || (n_out > 0 && (!names || !outs))) { fprintf(stderr, "[%s] cml_predict: bad outputs\n", m->fn); return -1; }
    for (int k = 0; k < n_out; k++)
        if (!names[k] || !outs[k]) { fprintf(stderr, "[%s] cml_predict: output %d is NULL\n", m->fn, k); return -1; }
    void *pool = rt_pool_push();
    int rc = run(m, ids, mask, disc, names, n_out, outs);
    rt_pool_pop(pool);
    return rc;
}

void cml_free(cml_model *m) {
    if (!m) return;
    if (rt_msg) {
        void *pool = rt_pool_push();
        for (int i = 0; i < m->n_in; i++) {
            release(m->in[i].array);
            release(m->in[i].name);
        }
        release(m->model);
        rt_pool_pop(pool);
    }
    free(m);
}
