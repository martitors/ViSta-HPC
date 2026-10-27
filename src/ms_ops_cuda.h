// =============================================================================
// ms_ops_cuda.h — interface between ms_ops.cpp and the GPU engine ms_ops_cuda.cu
// =============================================================================
//
//   VistaItem          : one block of rows of ONE MS (host pointers)
//   vista_geometry     : UVW rotation matrix + (dl, dm, dn) + 1/(1+z)
//   vista_median_df    : median |df| of a frequency grid
//   vista_df_old_k     : per-channel width of the input grid (centred diff.)
//   vista_build_map    : input -> output channel map with the overlap weights
//                        (CSR), built with exactly the CPU formulas
//   vista_process_channel : one output channel of one row (GPU kernel body)
//   vista_plan_passes  : splits the work into passes that fit in GPU memory
//
// The host functions use the same formulas as the CPU kernel in ms_ops.cpp,
// so CPU and GPU compute the geometry, the channel widths and the rebinning
// weights identically.  This header compiles without CUDA as well; the C API
// at the bottom exists only in CUDA builds (WITH_CUDA).
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// A block of rows of one MS.  All arrays are C-contiguous host memory.
// ---------------------------------------------------------------------------
struct VistaItem {
    const float*   vis_in;    // complex64, interleaved  [nrow, nchan_old, ncorr]
    const uint8_t* flag_in;   // bool (1 byte)           [nrow, nchan_old, ncorr]
    const double*  uvw_in;    //                         [nrow, 3]
    const double*  freq_old;  // rest-frame input freqs  [nchan_old]
    const double*  freq_new;  // output freqs            [nchan_new]
    int64_t nrow;
    int32_t nchan_old;
    int32_t nchan_new;
    int32_t ncorr;
    double  z, ra_old, dec_old, ra_new, dec_new;
    float*   vis_out;         // complex64, interleaved  [nrow, nchan_new, ncorr]
    uint8_t* flag_out;        // bool (1 byte)           [nrow, nchan_new, ncorr]
    double*  uvw_out;         //                         [nrow, 3]
    const double* width_old = nullptr;   // input channel widths [nchan_old], or nullptr
};

// ---------------------------------------------------------------------------
// Geometry: UVW rotation towards the new phase centre and the projection used
// by the phase shift.  Same formulas as the CPU kernel.
// ---------------------------------------------------------------------------
struct VistaGeometry {
    double inv_z;
    double dl, dm, dn;
    double R[9];
};

inline void vista_build_basis(double ra, double dec, double M[9])
{
    const double sr = std::sin(ra),  cr = std::cos(ra);
    const double sd = std::sin(dec), cd = std::cos(dec);
    M[0] = -sr;    M[1] =  cr;    M[2] = 0.0;
    M[3] = -cr*sd; M[4] = -sr*sd; M[5] = cd;
    M[6] =  cr*cd; M[7] =  sr*cd; M[8] = sd;
}

inline VistaGeometry vista_geometry(double z,
                                    double ra_old, double dec_old,
                                    double ra_new, double dec_new)
{
    VistaGeometry g;
    double bo[9], bn[9];
    vista_build_basis(ra_old, dec_old, bo);
    vista_build_basis(ra_new, dec_new, bn);
    // R = basis_new * basis_old^T
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += bn[i*3+k] * bo[j*3+k];
            g.R[i*3+j] = s;
        }
    g.dl = (ra_new - ra_old) * std::cos(dec_old);
    g.dm = dec_new - dec_old;
    const double arg = 1.0 - g.dl*g.dl - g.dm*g.dm;
    g.dn = (arg > 0.0 ? std::sqrt(arg) : 0.0) - 1.0;
    g.inv_z = 1.0 / (1.0 + z);
    return g;
}

// Median |df| of a grid.
inline double vista_median_df(const double* f, int64_t n)
{
    if (n < 2) return 1.0;
    std::vector<double> d(static_cast<size_t>(n - 1));
    for (int64_t i = 0; i < n - 1; ++i) d[i] = std::abs(f[i+1] - f[i]);
    std::nth_element(d.begin(), d.begin() + d.size()/2, d.end());
    return d[d.size()/2];
}

// Per-channel width of the input grid: centred difference, forward/backward
// difference at the edges, always positive (as CASA mstransform).
// When the real widths (CHAN_WIDTH) are given they are used instead: the
// centred difference is wrong at the junction of two SPWs separated by a gap.
inline void vista_df_old_k(const double* fo, int64_t nco,
                           const double* fn, int64_t ncn,
                           std::vector<double>& out,
                           const double* width = nullptr)
{
    if (width) {
        out.assign(width, width + nco);
        for (auto& v : out) v = std::abs(v);
        return;
    }
    out.assign(static_cast<size_t>(nco), 1.0);
    if (nco == 1) {
        out[0] = (ncn > 1) ? (fn[1] - fn[0]) : 1.0;
    } else {
        out[0] = fo[1] - fo[0];
        for (int64_t k = 1; k < nco - 1; ++k) out[k] = (fo[k+1] - fo[k-1]) * 0.5;
        out[nco-1] = fo[nco-1] - fo[nco-2];
    }
    for (auto& v : out) v = std::abs(v);
}

// ---------------------------------------------------------------------------
// Input -> output channel map in CSR form.  For each output channel j:
//   k_idx[jptr[j] .. jptr[j+1]-1] = input channels that overlap j
//   w    [jptr[j] .. jptr[j+1]-1] = overlap weight  ov / df_old_k[k]
// with k increasing (the CPU accumulation order).  The j0..j1 range of each k
// uses exactly the CPU formula, so the set of (j, k, w) is the same for any
// input grid (increasing, decreasing, concatenated SPWs with gaps, one channel).
// ---------------------------------------------------------------------------
inline void vista_build_map(const double* fo, int64_t nco,
                            const double* fn, int64_t ncn,
                            const std::vector<double>& df_old_k,
                            double df_new,
                            std::vector<int64_t>& jptr,   // [ncn + 1]
                            std::vector<int32_t>& k_idx,  // [nnz]
                            std::vector<double>&  w)      // [nnz]
{
    std::vector<int64_t> count(static_cast<size_t>(ncn), 0);
    auto j_range = [&](int64_t k, int64_t& j0, int64_t& j1) {
        const double lo_in = fo[k] - df_old_k[k] * 0.5;
        const double hi_in = fo[k] + df_old_k[k] * 0.5;
        // clamp in double before the cast (no overflow on far-away values)
        double a = std::floor((lo_in - (fn[0] + df_new * 0.5)) / df_new);
        double b = std::floor((hi_in - (fn[0] - df_new * 0.5)) / df_new);
        a = std::max(a, 0.0);
        b = std::min(b, static_cast<double>(ncn - 1));
        if (!(b >= a)) { j0 = 0; j1 = -1; return; }   // no overlap (or NaN)
        j0 = static_cast<int64_t>(a);
        j1 = static_cast<int64_t>(b);
    };
    auto weight = [&](int64_t k, int64_t j) {
        const double dfo    = df_old_k[k];
        const double lo_in  = fo[k] - dfo * 0.5,     hi_in  = fo[k] + dfo * 0.5;
        const double lo_out = fn[j] - df_new * 0.5,  hi_out = fn[j] + df_new * 0.5;
        const double ov = std::max(0.0, std::min(hi_in, hi_out) - std::max(lo_in, lo_out));
        return ov / dfo;
    };

    // first pass: count the pairs with weight > 0 for each j
    for (int64_t k = 0; k < nco; ++k) {
        int64_t j0, j1; j_range(k, j0, j1);
        for (int64_t j = j0; j <= j1; ++j)
            if (weight(k, j) > 0.0) ++count[j];
    }
    jptr.assign(static_cast<size_t>(ncn + 1), 0);
    for (int64_t j = 0; j < ncn; ++j) jptr[j+1] = jptr[j] + count[j];
    k_idx.assign(static_cast<size_t>(jptr[ncn]), 0);
    w.assign(static_cast<size_t>(jptr[ncn]), 0.0);

    // second pass: fill (k increasing => same accumulation order as the CPU)
    std::vector<int64_t> pos(jptr.begin(), jptr.end() - 1);
    for (int64_t k = 0; k < nco; ++k) {
        int64_t j0, j1; j_range(k, j0, j1);
        for (int64_t j = j0; j <= j1; ++j) {
            const double wk = weight(k, j);
            if (wk <= 0.0) continue;
            k_idx[pos[j]] = static_cast<int32_t>(k);
            w[pos[j]]     = wk;
            ++pos[j];
        }
    }
}

// ---------------------------------------------------------------------------
// One output channel of one row (body of the GPU kernel), __host__ __device__.
//   vis_in_row  : complex64 interleaved of the row  [nchan_old, ncorr, 2]
//   flag_in_row : flags of the row                  [nchan_old, ncorr]
//   vis_out_row : complex64 interleaved output       [nchan_new, ncorr, 2]
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define VISTA_HD __host__ __device__
#else
#define VISTA_HD
#endif

#define VISTA_MAX_CORR 4      // an MS has at most 4 correlations

VISTA_HD inline void vista_sincos(double x, double* s, double* c)
{
#ifdef __CUDA_ARCH__
    sincos(x, s, c);
#else
    *s = std::sin(x);
    *c = std::cos(x);
#endif
}

// NC = number of correlations, known at compile time: the accumulators stay
// in GPU registers instead of local memory.
template <int NC>
VISTA_HD inline void vista_process_channel_t(
    int j, double proj,
    const double*  fo,          // rest-frame input freqs [nchan_old]
    const int64_t* jptr,        // CSR map (see vista_build_map)
    const int32_t* kidx,
    const double*  wts,
    const float*   vis_in_row,
    const uint8_t* flag_in_row,
    float*         vis_out_row,
    uint8_t*       flag_out_row)
{
    const double C_LIGHT_ = 299792458.0;
    const double TWO_PI_  = 6.283185307179586;

    double ar[NC], ai[NC], ws[NC];
    for (int c = 0; c < NC; ++c) { ar[c] = 0.0; ai[c] = 0.0; ws[c] = 0.0; }

    for (int64_t p = jptr[j]; p < jptr[j+1]; ++p) {
        const int    k  = kidx[p];
        const double wk = wts[p];
        // phase in float64, same expression as the CPU
        const double phi = -TWO_PI_ * proj * fo[k] / C_LIGHT_;
        double sn, cs;
        vista_sincos(phi, &sn, &cs);
        const size_t idx = (size_t)k * NC;
        for (int c = 0; c < NC; ++c) {
            if (flag_in_row[idx + c]) continue;
            const double vr = (double)vis_in_row[2*(idx + c)];
            const double vi = (double)vis_in_row[2*(idx + c) + 1];
            ar[c] += (vr*cs - vi*sn) * wk;
            ai[c] += (vr*sn + vi*cs) * wk;
            ws[c] += wk;
        }
    }

    const size_t o = (size_t)j * NC;
    for (int c = 0; c < NC; ++c) {
        if (ws[c] > 0.0) {
            vis_out_row[2*(o + c)]     = (float)(ar[c] / ws[c]);
            vis_out_row[2*(o + c) + 1] = (float)(ai[c] / ws[c]);
            flag_out_row[o + c] = 0;
        } else {
            vis_out_row[2*(o + c)]     = 0.0f;
            vis_out_row[2*(o + c) + 1] = 0.0f;
            flag_out_row[o + c] = 1;
        }
    }
}

VISTA_HD inline void vista_process_channel(
    int j, double proj, int ncorr,
    const double* fo, const int64_t* jptr, const int32_t* kidx, const double* wts,
    const float* vis_in_row, const uint8_t* flag_in_row,
    float* vis_out_row, uint8_t* flag_out_row)
{
    switch (ncorr) {
    case 1: vista_process_channel_t<1>(j, proj, fo, jptr, kidx, wts, vis_in_row, flag_in_row, vis_out_row, flag_out_row); break;
    case 2: vista_process_channel_t<2>(j, proj, fo, jptr, kidx, wts, vis_in_row, flag_in_row, vis_out_row, flag_out_row); break;
    case 3: vista_process_channel_t<3>(j, proj, fo, jptr, kidx, wts, vis_in_row, flag_in_row, vis_out_row, flag_out_row); break;
    case 4: vista_process_channel_t<4>(j, proj, fo, jptr, kidx, wts, vis_in_row, flag_in_row, vis_out_row, flag_out_row); break;
    default: break;   // invalid ncorr: rejected on the host before the launch
    }
}

// ---------------------------------------------------------------------------
// Split the work into passes that fit in 'budget' bytes of GPU memory.  A pass
// holds segments = consecutive rows of one block; a large block may be split
// across passes.
// ---------------------------------------------------------------------------
struct VistaSegment {
    int     item;
    int64_t row0, nrow;          // rows of the block
    int64_t in_off, out_off;     // element offsets in the pass buffers
    int64_t row_begin;           // first row in the numbering of the pass
};

struct VistaPass {
    std::vector<VistaSegment> segs;
    int64_t rows = 0;
    size_t  in_elems = 0, out_elems = 0;
};

// GPU bytes per row: input (vis 8 B + flag 1 B), output (vis 8 B + flag 1 B),
// UVW in/out (2 x 24 B).
inline size_t vista_bytes_per_row(const VistaItem& it)
{
    return (size_t)it.nchan_old * it.ncorr * 9
         + (size_t)it.nchan_new * it.ncorr * 9
         + 48;
}

// Returns false if the budget is too small even for a single row.
inline bool vista_plan_passes(const VistaItem* items, int n_items, size_t budget,
                              int64_t max_rows_per_pass,
                              std::vector<VistaPass>& passes)
{
    passes.assign(1, VistaPass());
    size_t used = 0;
    for (int m = 0; m < n_items; ++m) {
        const VistaItem& it = items[m];
        const size_t bpr = vista_bytes_per_row(it);
        int64_t row = 0;
        while (row < it.nrow) {
            VistaPass& p = passes.back();
            int64_t fit = (int64_t)((budget > used ? budget - used : 0) / bpr);
            fit = std::min(fit, max_rows_per_pass - p.rows);
            if (fit <= 0) {
                if (p.rows == 0) return false;
                passes.emplace_back();       // pass full: open a new one
                used = 0;
                continue;
            }
            const int64_t take = std::min<int64_t>(fit, it.nrow - row);
            VistaSegment s;
            s.item      = m;
            s.row0      = row;
            s.nrow      = take;
            s.in_off    = (int64_t)p.in_elems;
            s.out_off   = (int64_t)p.out_elems;
            s.row_begin = p.rows;
            p.segs.push_back(s);
            p.rows      += take;
            p.in_elems  += (size_t)take * it.nchan_old * it.ncorr;
            p.out_elems += (size_t)take * it.nchan_new * it.ncorr;
            used        += (size_t)take * bpr;
            row         += take;
        }
    }
    if (passes.back().rows == 0) passes.pop_back();
    return true;
}

// ---------------------------------------------------------------------------
// GPU C API (implemented in ms_ops_cuda.cu, CUDA builds only)
// ---------------------------------------------------------------------------
#ifdef WITH_CUDA
extern "C" {
// Number of visible GPUs (0 if none or no driver).
int vista_cuda_device_count();
// Free / total memory of the current GPU in bytes.  Returns 0 on success.
int vista_cuda_mem_info(size_t* free_bytes, size_t* total_bytes);
// Process n_items blocks (of any size) on the GPU.  If the data do not fit in
// the fraction mem_fraction of the free memory, the work is split into passes.
// Returns 0 on success, otherwise writes the error message into err.
int vista_cuda_run(const VistaItem* items, int n_items, double mem_fraction,
                   char* err, size_t err_len);
// Page-lock (pin) / unlock host memory that is already allocated.  0 = OK.
int vista_cuda_host_register(void* ptr, size_t bytes);
int vista_cuda_host_unregister(void* ptr);
}
#endif
