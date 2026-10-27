// =============================================================================
// ms_ops_cuda.cu — GPU engine of full_pipeline_batch_into (ms_ops.cpp)
// =============================================================================
//
// Same physics as the CPU kernel in ms_ops.cpp:
//   - rest-frame UVW (x 1/(1+z)) and rotation towards the new phase centre
//   - phase shift on the rest-frame grid, in float64
//   - overlap-weighted rebinning with the per-channel input width
//
// How the work is organised:
//   - the input -> output channel map (indices + weights, CSR) is built on the
//     host with the CPU formulas (vista_build_map), so decreasing SPWs and
//     concatenated SPWs with gaps are handled exactly as on the CPU;
//   - one call takes many blocks (VistaItem) of different shapes, typically
//     chunks of different MSs; the work is split automatically into passes
//     that fit in the free GPU memory;
//   - one CUDA block per row, its threads over the output channels: neighbour
//     threads touch neighbour channels of the same row (coalesced accesses);
//     no complex128 scratch buffer, the phase is computed on the fly;
//   - the large pass buffers stay allocated between calls and only grow
//     (cudaMalloc/cudaFree are slow and synchronise the device);
//   - the outputs are copied straight into the caller's arrays; when those are
//     page-locked (vista_cuda_host_register) the copies are direct DMA;
//   - all indices on the large arrays are 64-bit; all GPU memory of a call is
//     released by RAII, also on errors.
//
// The phase is evaluated with sincos for every channel instead of the CPU
// recurrence, so a few output values may differ from the CPU in the last
// float32 digit.  Compile without --use_fast_math.
//
// C API (declared in ms_ops_cuda.h):
//   int vista_cuda_device_count()
//   int vista_cuda_mem_info(size_t* free, size_t* total)
//   int vista_cuda_run(const VistaItem*, int, double, char*, size_t)
//   int vista_cuda_host_register(void*, size_t) / vista_cuda_host_unregister(void*)
// =============================================================================

#include "ms_ops_cuda.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int THREADS = 128;                    // threads per block (= per row)

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess)                                                     \
        throw std::runtime_error(std::string("CUDA error: ")                   \
            + cudaGetErrorString(_e) + " at " __FILE__ ":"                     \
            + std::to_string(__LINE__));                                       \
} while (0)

// GPU buffer released by its destructor (also when an exception is thrown).
template <class T>
class DevBuf {
public:
    DevBuf() = default;
    explicit DevBuf(size_t n) { alloc(n); }
    ~DevBuf() { release(); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    void alloc(size_t n) {
        release();
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)));
    }
    void release() {
        if (p_) cudaFree(p_);
        p_ = nullptr;
    }
    void upload(const T* src, size_t n, size_t offset = 0) {
        if (n) CUDA_CHECK(cudaMemcpy(p_ + offset, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    T* get() const { return p_; }

private:
    T* p_ = nullptr;
};

// Persistent GPU buffer that only grows: kept for the life of the process.
template <class T>
class GrowBuf {
public:
    T* ensure(size_t n) {
        if (n > cap_) {
            if (p_) cudaFree(p_);
            p_ = nullptr; cap_ = 0;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)));
            cap_ = n;
        }
        return p_;
    }
    size_t bytes() const { return cap_ * sizeof(T); }
private:
    T* p_ = nullptr;
    size_t cap_ = 0;
};

struct PassPool {
    GrowBuf<float>   vis_in, vis_out;
    GrowBuf<uint8_t> flag_in, flag_out;
    GrowBuf<double>  uvw_in, uvw_out;
    size_t bytes() const {
        return vis_in.bytes() + vis_out.bytes() + flag_in.bytes() + flag_out.bytes()
             + uvw_in.bytes() + uvw_out.bytes();
    }
};
static PassPool g_pool;

inline void h2d(void* dst, const void* src, size_t bytes) {
    if (bytes) CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
inline void d2h(void* dst, const void* src, size_t bytes) {
    if (bytes) CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

// Per-block parameters copied to the GPU
struct ItemDev {
    double  inv_z, dl, dm, dn;
    double  R[9];
    int64_t fo_off;     // offset in fo_all   (input frequencies)
    int64_t jptr_off;   // offset in jptr_all (nchan_new + 1 elements)
    int32_t nchan_old, nchan_new, ncorr, pad;
};

// Segment = consecutive rows of one block loaded in one pass
struct SegDev {
    int64_t in_off;     // element offset in vis_in / flag_in of the pass
    int64_t out_off;    // element offset in vis_out / flag_out of the pass
    int64_t row_begin;  // first row of the segment in the numbering of the pass
    int32_t item;       // block index
    int32_t pad;
};

// One CUDA block per row; its threads share the output channels.
__global__ void vista_kernel(
    const float*   __restrict__ vis_in,      // complex64 interleaved
    const uint8_t* __restrict__ flag_in,
    const double*  __restrict__ uvw_in,      // [rows_pass, 3]
    const double*  __restrict__ fo_all,
    const int64_t* __restrict__ jptr_all,
    const int32_t* __restrict__ kidx_all,
    const double*  __restrict__ w_all,
    const ItemDev* __restrict__ items,
    const SegDev*  __restrict__ segs,
    int n_segs,
    float*   __restrict__ vis_out,           // complex64 interleaved
    uint8_t* __restrict__ flag_out,
    double*  __restrict__ uvw_out)           // [rows_pass, 3]
{
    const int64_t grow = blockIdx.x;         // row in the pass

    // segment holding this row (binary search, n_segs is small)
    int lo = 0, hi = n_segs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (segs[mid].row_begin <= grow) lo = mid; else hi = mid - 1;
    }
    const SegDev  s  = segs[lo];
    const ItemDev it = items[s.item];
    const int64_t r  = grow - s.row_begin;   // row inside the segment
    const int nco = it.nchan_old, ncn = it.nchan_new, ncr = it.ncorr;

    const size_t in_base  = (size_t)s.in_off  + (size_t)r * nco * ncr;
    const size_t out_base = (size_t)s.out_off + (size_t)r * ncn * ncr;

    // rest-frame UVW and rotation
    const double u = uvw_in[grow*3 + 0] * it.inv_z;
    const double v = uvw_in[grow*3 + 1] * it.inv_z;
    const double w = uvw_in[grow*3 + 2] * it.inv_z;
    if (threadIdx.x == 0) {
        uvw_out[grow*3 + 0] = it.R[0]*u + it.R[1]*v + it.R[2]*w;
        uvw_out[grow*3 + 1] = it.R[3]*u + it.R[4]*v + it.R[5]*w;
        uvw_out[grow*3 + 2] = it.R[6]*u + it.R[7]*v + it.R[8]*w;
    }
    const double proj = u*it.dl + v*it.dm + w*it.dn;

    // phase shift + rebinning, one output channel per thread
    for (int j = threadIdx.x; j < ncn; j += blockDim.x)
        vista_process_channel(j, proj, ncr,
                              fo_all + it.fo_off, jptr_all + it.jptr_off, kidx_all, w_all,
                              vis_in + 2 * in_base, flag_in + in_base,
                              vis_out + 2 * out_base, flag_out + out_base);
}

// Whole run (throws; vista_cuda_run turns the exceptions into an error code)
void run_impl(const VistaItem* items, int n_items, double mem_fraction)
{
    if (n_items <= 0) return;
    if (!(mem_fraction > 0.0 && mem_fraction <= 1.0)) mem_fraction = 0.85;

    // --- 1. parameters and channel map of every block (host) ---
    std::vector<ItemDev> item_dev(n_items);
    std::vector<double>  fo_all;
    std::vector<int64_t> jptr_all;
    std::vector<int32_t> kidx_all;
    std::vector<double>  w_all;
    std::vector<double>  df_old_k;
    std::vector<int64_t> jptr;
    std::vector<int32_t> kidx;
    std::vector<double>  wts;

    for (int m = 0; m < n_items; ++m) {
        const VistaItem& it = items[m];
        if (it.ncorr < 1 || it.ncorr > VISTA_MAX_CORR)
            throw std::runtime_error("unsupported number of correlations: " + std::to_string(it.ncorr));
        if (it.nchan_old < 1 || it.nchan_new < 1)
            throw std::runtime_error("empty frequency grid");

        const VistaGeometry g = vista_geometry(it.z, it.ra_old, it.dec_old,
                                               it.ra_new, it.dec_new);
        vista_df_old_k(it.freq_old, it.nchan_old, it.freq_new, it.nchan_new, df_old_k,
                       it.width_old);
        const double df_new = vista_median_df(it.freq_new, it.nchan_new);
        vista_build_map(it.freq_old, it.nchan_old, it.freq_new, it.nchan_new,
                        df_old_k, df_new, jptr, kidx, wts);

        ItemDev& d = item_dev[m];
        d.inv_z = g.inv_z; d.dl = g.dl; d.dm = g.dm; d.dn = g.dn;
        for (int q = 0; q < 9; ++q) d.R[q] = g.R[q];
        d.fo_off    = (int64_t)fo_all.size();
        d.jptr_off  = (int64_t)jptr_all.size();
        d.nchan_old = it.nchan_old;
        d.nchan_new = it.nchan_new;
        d.ncorr     = it.ncorr;
        d.pad       = 0;

        fo_all.insert(fo_all.end(), it.freq_old, it.freq_old + it.nchan_old);
        const int64_t base = (int64_t)kidx_all.size();   // absolute jptr
        for (int64_t x : jptr) jptr_all.push_back(base + x);
        kidx_all.insert(kidx_all.end(), kidx.begin(), kidx.end());
        w_all.insert(w_all.end(), wts.begin(), wts.end());
    }

    // --- 2. small arrays shared by all passes ---
    DevBuf<ItemDev> d_items(item_dev.size());   d_items.upload(item_dev.data(), item_dev.size());
    DevBuf<double>  d_fo(fo_all.size());        d_fo.upload(fo_all.data(), fo_all.size());
    DevBuf<int64_t> d_jptr(jptr_all.size());    d_jptr.upload(jptr_all.data(), jptr_all.size());
    DevBuf<int32_t> d_kidx(kidx_all.size());    d_kidx.upload(kidx_all.data(), kidx_all.size());
    DevBuf<double>  d_w(w_all.size());          d_w.upload(w_all.data(), w_all.size());

    // --- 3. passes that fit in the free memory ---
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    // the memory already held by the persistent pool can be reused
    const size_t budget = (size_t)((double)(free_b + g_pool.bytes()) * mem_fraction);
    std::vector<VistaPass> passes;
    if (!vista_plan_passes(items, n_items, budget, (int64_t)INT_MAX, passes))
        throw std::runtime_error("not enough GPU memory even for a single row");

    // --- 4. for every pass: copy in, compute, copy out ---
    for (const VistaPass& p : passes) {
        float*   d_vis_in   = g_pool.vis_in.ensure(2 * p.in_elems);     // complex64
        float*   d_vis_out  = g_pool.vis_out.ensure(2 * p.out_elems);
        uint8_t* d_flag_in  = g_pool.flag_in.ensure(p.in_elems);
        uint8_t* d_flag_out = g_pool.flag_out.ensure(p.out_elems);
        double*  d_uvw_in   = g_pool.uvw_in.ensure((size_t)p.rows * 3);
        double*  d_uvw_out  = g_pool.uvw_out.ensure((size_t)p.rows * 3);

        std::vector<SegDev> seg_dev(p.segs.size());
        for (size_t q = 0; q < p.segs.size(); ++q) {
            const VistaSegment& s = p.segs[q];
            const VistaItem& it = items[s.item];
            const size_t row_in  = (size_t)it.nchan_old * it.ncorr;   // elements per row
            const size_t n_in    = (size_t)s.nrow * row_in;

            h2d(d_vis_in + 2 * (size_t)s.in_off, it.vis_in + 2 * (size_t)s.row0 * row_in,
                2 * n_in * sizeof(float));
            h2d(d_flag_in + (size_t)s.in_off, it.flag_in + (size_t)s.row0 * row_in, n_in);
            h2d(d_uvw_in + (size_t)s.row_begin * 3, it.uvw_in + (size_t)s.row0 * 3,
                (size_t)s.nrow * 3 * sizeof(double));

            seg_dev[q] = SegDev{ s.in_off, s.out_off, s.row_begin, s.item, 0 };
        }
        DevBuf<SegDev> d_segs(seg_dev.size());
        d_segs.upload(seg_dev.data(), seg_dev.size());

        vista_kernel<<<(unsigned int)p.rows, THREADS>>>(
            d_vis_in, d_flag_in, d_uvw_in,
            d_fo.get(), d_jptr.get(), d_kidx.get(), d_w.get(),
            d_items.get(), d_segs.get(), (int)seg_dev.size(),
            d_vis_out, d_flag_out, d_uvw_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        for (const VistaSegment& s : p.segs) {
            const VistaItem& it = items[s.item];
            const size_t row_out = (size_t)it.nchan_new * it.ncorr;
            const size_t n_out   = (size_t)s.nrow * row_out;
            d2h(it.vis_out + 2 * (size_t)s.row0 * row_out, d_vis_out + 2 * (size_t)s.out_off,
                2 * n_out * sizeof(float));
            d2h(it.flag_out + (size_t)s.row0 * row_out, d_flag_out + (size_t)s.out_off, n_out);
            d2h(it.uvw_out + (size_t)s.row0 * 3, d_uvw_out + (size_t)s.row_begin * 3,
                (size_t)s.nrow * 3 * sizeof(double));
        }
    }
}

}  // namespace


// =============================================================================
// C API
// =============================================================================
extern "C" {

int vista_cuda_device_count()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();   // clear the error
        return 0;
    }
    return n;
}

int vista_cuda_mem_info(size_t* free_bytes, size_t* total_bytes)
{
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        cudaGetLastError();
        if (free_bytes)  *free_bytes = 0;
        if (total_bytes) *total_bytes = 0;
        return 1;
    }
    if (free_bytes)  *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return 0;
}

int vista_cuda_host_register(void* ptr, size_t bytes)
{
    if (!ptr || !bytes) return 0;
    if (cudaHostRegister(ptr, bytes, cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return 0;
}

int vista_cuda_host_unregister(void* ptr)
{
    if (!ptr) return 0;
    if (cudaHostUnregister(ptr) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return 0;
}

int vista_cuda_run(const VistaItem* items, int n_items, double mem_fraction,
                   char* err, size_t err_len)
{
    try {
        run_impl(items, n_items, mem_fraction);
        return 0;
    } catch (const std::exception& e) {
        if (err && err_len) std::snprintf(err, err_len, "%s", e.what());
    } catch (...) {
        if (err && err_len) std::snprintf(err, err_len, "unknown error");
    }
    return 1;
}

}  // extern "C"
