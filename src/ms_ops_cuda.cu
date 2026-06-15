// =============================================================================
// ms_ops_cuda.cu
// =============================================================================
//
// CUDA implementation of the ViSta visibility processing kernel.
// Provides GPU-accelerated versions of full_pipeline and full_pipeline_batch,
// called from ms_ops.cpp when compiled with -DWITH_CUDA.
//
// The algorithm is identical to the OpenMP C++ implementation in ms_ops.cpp:
//
//   FIX A: Input visibilities are complex64 (float2). Each element is cast
//          to float64 inline, before the phase multiplication, to maintain
//          double-precision accuracy for the phase computation.
//          (Baseline lengths ~10 km * frequencies ~700 GHz require double
//          precision; accumulating phase errors would decorrelate the stack.)
//
//   FIX B: The spectral rebinning uses a per-channel input width df_old_k[k]
//          computed on the host as a centred finite difference, rather than a
//          global median.  This matches CASA's mstransform exactly.
//
//   FIX C: The weight scale factor R = df_new / df_old_obs is computed in
//          Python and applied there; it is not part of this kernel.
//
// Thread organisation:
//   full_pipeline_kernel       : one CUDA thread per baseline row.
//   full_pipeline_batch_kernel : one CUDA thread per (MS index, baseline row).
//
// Both kernels use a global scratch buffer (cuDoubleComplex) to store the
// phase-shifted visibilities between step 2 (phase shift) and step 3
// (rebinning).  The scratch buffer is allocated on the GPU for each call and
// freed before returning.
//
// External C API (called from ms_ops.cpp):
//   int  cuda_available()
//   void full_pipeline_cuda(...)
//   void full_pipeline_batch_cuda(...)
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr double C_LIGHT = 299792458.0;  // speed of light [m/s]
static constexpr double TWO_PI  = 6.283185307179586;


// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
// Wraps every CUDA call and throws a std::runtime_error if it fails,
// including the file name and line number for easy debugging.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") \
            + cudaGetErrorString(_err) \
            + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
} while(0)


// ===========================================================================
// full_pipeline_kernel
// ===========================================================================
// Single-MS CUDA kernel.  One thread is launched per baseline row; the
// thread applies all three pipeline steps to its row independently.
//
// Parameters
// ----------
//   vis_in_f32        : complex64 input  [nrow, nchan_old, ncorr]
//   flag_in           : bool flags       [nrow, nchan_old, ncorr]
//   uvw_in            : UVW coords       [nrow, 3]
//   freq_old_rf       : rest-framed input channel centres [nchan_old]
//   freq_new          : output channel centres            [nchan_new]
//   df_old_k          : per-channel input widths [nchan_old]  (FIX B)
//   inv_z             : 1 / (1+z)
//   dl, dm, dn        : direction-cosine differences (phase shift)
//   Rot0..Rot8        : 3x3 UVW rotation matrix (flattened row-major)
//   df_new            : output channel width
//   freq_old0         : first input channel centre (recurrence starting point)
//   df_old_eq         : input channel spacing (recurrence step, equispaced only)
//   equispaced        : 1 if input grid is equispaced (enables recurrence)
//   vis_out, flag_out, uvw_out : output arrays
//   vis_shifted_global : global scratch [nrow, nchan_old, ncorr] cuDoubleComplex
// ===========================================================================
__global__ void full_pipeline_kernel(
    const float2*          __restrict__ vis_in_f32,
    const uint8_t*         __restrict__ flag_in,
    const double*          __restrict__ uvw_in,
    const double*          __restrict__ freq_old_rf,
    const double*          __restrict__ freq_new,
    const double*          __restrict__ df_old_k,
    int    nrow, int nchan_old, int nchan_new, int ncorr,
    double inv_z,
    double dl, double dm, double dn,
    double Rot0, double Rot1, double Rot2,
    double Rot3, double Rot4, double Rot5,
    double Rot6, double Rot7, double Rot8,
    double df_new,
    double freq_old0,
    double df_old_eq,
    int    equispaced,
    float2*  __restrict__ vis_out,
    uint8_t* __restrict__ flag_out,
    double*  __restrict__ uvw_out,
    cuDoubleComplex* __restrict__ vis_shifted_global
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) return;

    // -----------------------------------------------------------------------
    // Step 1: rest-frame UVW (scale by 1/(1+z)) and rotate to new phase centre
    // -----------------------------------------------------------------------
    double u_rf = uvw_in[i*3+0] * inv_z;
    double v_rf = uvw_in[i*3+1] * inv_z;
    double w_rf = uvw_in[i*3+2] * inv_z;

    uvw_out[i*3+0] = Rot0*u_rf + Rot1*v_rf + Rot2*w_rf;
    uvw_out[i*3+1] = Rot3*u_rf + Rot4*v_rf + Rot5*w_rf;
    uvw_out[i*3+2] = Rot6*u_rf + Rot7*v_rf + Rot8*w_rf;

    double proj = u_rf*dl + v_rf*dm + w_rf*dn;

    // Scratch pointer for this row's phase-shifted visibilities
    cuDoubleComplex* vis_sh = vis_shifted_global + (size_t)i * nchan_old * ncorr;

    // -----------------------------------------------------------------------
    // Step 2: phase shift on the rest-framed input grid
    // FIX A: cast float32 -> float64 inline, before the multiplication.
    // Recurrence phasor (refreshed every 64 channels for numerical stability).
    // -----------------------------------------------------------------------
    if (equispaced && nchan_old > 1) {
        double phi0 = -TWO_PI * proj * freq_old0 / C_LIGHT;
        double dphi = -TWO_PI * proj * df_old_eq / C_LIGHT;
        for (int j = 0; j < nchan_old; ++j) {
            // Recompute exact phase every 64 channels to prevent drift
            double phi = phi0 + (double)j * dphi;
            double cos_phi, sin_phi;
            sincos(phi, &sin_phi, &cos_phi);
            for (int c = 0; c < ncorr; ++c) {
                float2 vf = vis_in_f32[(i * nchan_old + j) * ncorr + c];
                double vr = (double)vf.x;
                double vi = (double)vf.y;
                vis_sh[j * ncorr + c] = make_cuDoubleComplex(
                    vr * cos_phi - vi * sin_phi,
                    vr * sin_phi + vi * cos_phi
                );
            }
        }
    } else {
        // Non-equispaced grid: compute exact phase per channel
        for (int j = 0; j < nchan_old; ++j) {
            double phi = -TWO_PI * proj * freq_old_rf[j] / C_LIGHT;
            double cos_phi, sin_phi;
            sincos(phi, &sin_phi, &cos_phi);
            for (int c = 0; c < ncorr; ++c) {
                float2 vf = vis_in_f32[(i * nchan_old + j) * ncorr + c];
                double vr = (double)vf.x;
                double vi = (double)vf.y;
                vis_sh[j * ncorr + c] = make_cuDoubleComplex(
                    vr * cos_phi - vi * sin_phi,
                    vr * sin_phi + vi * cos_phi
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: overlap-weighted spectral rebinning (output-centric loop)
    // FIX B: use per-channel df_old_k[k] as the normalisation width.
    //
    // The output-centric loop (iterating over output channels j first, then
    // input channels k) is more cache-friendly on GPU than the input-centric
    // loop used in the C++ version, because consecutive threads write to
    // consecutive output locations.
    //
    // k-range narrowing: a coarse estimate based on the approximate input
    // channel spacing restricts the k loop to the channels that can overlap
    // with output channel j, avoiding unnecessary iterations.
    // -----------------------------------------------------------------------
    double df_approx = (nchan_old > 1) ? fabs(freq_old_rf[1] - freq_old_rf[0]) : 1.0;

    for (int c = 0; c < ncorr; ++c) {
        for (int j = 0; j < nchan_new; ++j) {
            double accum_r = 0.0, accum_i = 0.0, out_wgt = 0.0;

            double lo_out = freq_new[j] - df_new * 0.5;
            double hi_out = freq_new[j] + df_new * 0.5;

            // Coarse k-range: restrict to channels that can overlap with [lo_out, hi_out]
            int k0 = max(0,           (int)floor((lo_out - freq_old_rf[0] - df_approx * 0.5) / df_approx) - 1);
            int k1 = min(nchan_old-1, (int)floor((hi_out - freq_old_rf[0] + df_approx * 0.5) / df_approx) + 1);

            for (int k = k0; k <= k1; ++k) {
                if (flag_in[(i * nchan_old + k) * ncorr + c]) continue;

                // FIX B: use per-channel width for exact overlap computation
                double dfo   = df_old_k[k];
                double lo_in = freq_old_rf[k] - dfo * 0.5;
                double hi_in = freq_old_rf[k] + dfo * 0.5;

                double ov = fmax(0.0, fmin(hi_in, hi_out) - fmax(lo_in, lo_out));
                double w  = ov / dfo;
                if (w <= 0.0) continue;

                cuDoubleComplex vs = vis_sh[k * ncorr + c];
                accum_r += cuCreal(vs) * w;
                accum_i += cuCimag(vs) * w;
                out_wgt += w;
            }

            int out_idx = (i * nchan_new + j) * ncorr + c;
            if (out_wgt > 0.0) {
                // Cast back to complex64 for output
                vis_out[out_idx]  = make_float2((float)(accum_r / out_wgt),
                                                (float)(accum_i / out_wgt));
                flag_out[out_idx] = 0;
            } else {
                vis_out[out_idx]  = make_float2(0.0f, 0.0f);
                flag_out[out_idx] = 1;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// build_basis_host
// ---------------------------------------------------------------------------
// Host-side helper to build the 3x3 orthonormal basis of the local tangent
// plane at (ra, dec).  Used to compute the UVW rotation matrix before the
// kernel launch.  Identical to the build_basis() function in ms_ops.cpp.
// ---------------------------------------------------------------------------
static void build_basis_host(double ra, double dec, double M[9])
{
    double sr = sin(ra), cr = cos(ra), sd = sin(dec), cd = cos(dec);
    M[0] = -sr;    M[1] =  cr;    M[2] = 0.0;
    M[3] = -cr*sd; M[4] = -sr*sd; M[5] = cd;
    M[6] =  cr*cd; M[7] =  sr*cd; M[8] = sd;
}


// ===========================================================================
// External C API
// ===========================================================================
extern "C" {


// ---------------------------------------------------------------------------
// cuda_available
// ---------------------------------------------------------------------------
// Check whether at least one CUDA device is present and accessible.
// Called from ms_ops.cpp at module load time to set _gpu_available.
// Returns 1 if a device is present, 0 otherwise.
// ---------------------------------------------------------------------------
int cuda_available()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count > 0 ? 1 : 0;
}


// ---------------------------------------------------------------------------
// full_pipeline_cuda
// ---------------------------------------------------------------------------
// Single-MS GPU entry point.  Called from ms_ops.cpp::full_pipeline() when
// _gpu_available is true.
//
// Workflow:
//   1. Compute geometric parameters (rotation matrix, direction cosines,
//      per-channel widths) on the host.
//   2. Allocate GPU buffers and copy input data host -> device.
//   3. Launch full_pipeline_kernel (one thread per row).
//   4. Copy results device -> host.
//   5. Free GPU buffers.
//
// All input/output arrays are in C-order (row-major), matching the layout
// produced by NumPy / pybind11.
// ---------------------------------------------------------------------------
void full_pipeline_cuda(
    const float*    vis_in_h,     // complex64: [nrow, nchan_old, ncorr] stored as float pairs
    const uint8_t*  flag_in_h,    // [nrow, nchan_old, ncorr]
    const double*   uvw_in_h,     // [nrow, 3]
    const double*   freq_old_h,   // [nchan_old]  rest-framed input frequencies
    const double*   freq_new_h,   // [nchan_new]  output frequencies
    int    nrow, int nchan_old, int nchan_new, int ncorr,
    double z,
    double ra_old, double dec_old,
    double ra_new, double dec_new,
    float*   vis_out_h,           // complex64: [nrow, nchan_new, ncorr]
    uint8_t* flag_out_h,          // [nrow, nchan_new, ncorr]
    double*  uvw_out_h            // [nrow, 3]
)
{
    // --- Geometric parameters (host) ---
    double basis_old[9], basis_new[9], Rot[9];
    build_basis_host(ra_old, dec_old, basis_old);
    build_basis_host(ra_new, dec_new, basis_new);
    // R = basis_new * basis_old^T
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Rot[i*3+j] = 0.0;
            for (int k = 0; k < 3; ++k)
                Rot[i*3+j] += basis_new[i*3+k] * basis_old[j*3+k];
        }
    double dl  = (ra_new - ra_old) * cos(dec_old);
    double dm  = dec_new - dec_old;
    double arg = 1.0 - dl*dl - dm*dm;
    double dn  = (arg > 0.0 ? sqrt(arg) : 0.0) - 1.0;
    double inv_z = 1.0 / (1.0 + z);

    // --- Per-channel input widths (FIX B, computed on host) ---
    std::vector<double> df_old_k_h(nchan_old);
    if (nchan_old == 1) {
        df_old_k_h[0] = (nchan_new > 1) ? fabs(freq_new_h[1] - freq_new_h[0]) : 1.0;
    } else {
        df_old_k_h[0] = freq_old_h[1] - freq_old_h[0];
        for (int k = 1; k < nchan_old - 1; ++k)
            df_old_k_h[k] = (freq_old_h[k+1] - freq_old_h[k-1]) * 0.5;
        df_old_k_h[nchan_old-1] = freq_old_h[nchan_old-1] - freq_old_h[nchan_old-2];
    }
    for (int k = 0; k < nchan_old; ++k)
        df_old_k_h[k] = fabs(df_old_k_h[k]);

    double df_new_v = (nchan_new > 1) ? fabs(freq_new_h[1] - freq_new_h[0]) : 1.0;

    // Equispaced check (enables recurrence in the kernel)
    int equispaced = 1;
    double df_old_eq = 0.0;
    if (nchan_old > 1) {
        df_old_eq = freq_old_h[1] - freq_old_h[0];
        for (int j = 2; j < nchan_old && equispaced; ++j)
            if (fabs((freq_old_h[j]-freq_old_h[j-1])-df_old_eq) > fabs(df_old_eq)*1e-6)
                equispaced = 0;
    }

    // --- GPU buffer sizes ---
    size_t sz_vis_in   = (size_t)nrow * nchan_old * ncorr * sizeof(float2);
    size_t sz_flag_in  = (size_t)nrow * nchan_old * ncorr;
    size_t sz_uvw_in   = (size_t)nrow * 3 * sizeof(double);
    size_t sz_freq_old = (size_t)nchan_old * sizeof(double);
    size_t sz_freq_new = (size_t)nchan_new * sizeof(double);
    size_t sz_df_old_k = (size_t)nchan_old * sizeof(double);
    size_t sz_vis_out  = (size_t)nrow * nchan_new * ncorr * sizeof(float2);
    size_t sz_flag_out = (size_t)nrow * nchan_new * ncorr;
    size_t sz_uvw_out  = (size_t)nrow * 3 * sizeof(double);
    // Scratch: double-precision phase-shifted visibilities
    size_t sz_vis_sh   = (size_t)nrow * nchan_old * ncorr * sizeof(cuDoubleComplex);

    // --- Allocate GPU buffers ---
    float2          *d_vis_in,  *d_vis_out;
    uint8_t         *d_flag_in, *d_flag_out;
    double          *d_uvw_in,  *d_freq_old, *d_freq_new, *d_df_old_k, *d_uvw_out;
    cuDoubleComplex *d_vis_sh;

    CUDA_CHECK(cudaMalloc(&d_vis_in,   sz_vis_in));
    CUDA_CHECK(cudaMalloc(&d_flag_in,  sz_flag_in));
    CUDA_CHECK(cudaMalloc(&d_uvw_in,   sz_uvw_in));
    CUDA_CHECK(cudaMalloc(&d_freq_old, sz_freq_old));
    CUDA_CHECK(cudaMalloc(&d_freq_new, sz_freq_new));
    CUDA_CHECK(cudaMalloc(&d_df_old_k, sz_df_old_k));
    CUDA_CHECK(cudaMalloc(&d_vis_out,  sz_vis_out));
    CUDA_CHECK(cudaMalloc(&d_flag_out, sz_flag_out));
    CUDA_CHECK(cudaMalloc(&d_uvw_out,  sz_uvw_out));
    CUDA_CHECK(cudaMalloc(&d_vis_sh,   sz_vis_sh));

    // --- Copy inputs host -> device ---
    CUDA_CHECK(cudaMemcpy(d_vis_in,   vis_in_h,           sz_vis_in,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flag_in,  flag_in_h,          sz_flag_in,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uvw_in,   uvw_in_h,           sz_uvw_in,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_freq_old, freq_old_h,         sz_freq_old, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_freq_new, freq_new_h,         sz_freq_new, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_df_old_k, df_old_k_h.data(),  sz_df_old_k, cudaMemcpyHostToDevice));

    // --- Kernel launch: 128 threads/block balances occupancy and register use ---
    int threads = 128;
    int blocks  = (nrow + threads - 1) / threads;

    full_pipeline_kernel<<<blocks, threads>>>(
        d_vis_in, d_flag_in, d_uvw_in,
        d_freq_old, d_freq_new, d_df_old_k,
        nrow, nchan_old, nchan_new, ncorr,
        inv_z, dl, dm, dn,
        Rot[0], Rot[1], Rot[2],
        Rot[3], Rot[4], Rot[5],
        Rot[6], Rot[7], Rot[8],
        df_new_v,
        freq_old_h[0], df_old_eq, equispaced,
        d_vis_out, d_flag_out, d_uvw_out, d_vis_sh
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy results device -> host ---
    CUDA_CHECK(cudaMemcpy(vis_out_h,  d_vis_out,  sz_vis_out,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flag_out_h, d_flag_out, sz_flag_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uvw_out_h,  d_uvw_out,  sz_uvw_out,  cudaMemcpyDeviceToHost));

    // --- Free GPU buffers ---
    cudaFree(d_vis_in);  cudaFree(d_flag_in);  cudaFree(d_uvw_in);
    cudaFree(d_freq_old); cudaFree(d_freq_new); cudaFree(d_df_old_k);
    cudaFree(d_vis_out); cudaFree(d_flag_out);  cudaFree(d_uvw_out);
    cudaFree(d_vis_sh);
}


// ===========================================================================
// full_pipeline_batch_kernel
// ===========================================================================
// Batch CUDA kernel.  One thread per (MS index, baseline row) pair.
// Processes n_ms Measurement Sets concurrently in a single kernel launch,
// sharing geometric parameters and frequency grids uploaded together.
//
// Buffer layout (all flat C-order, MS concatenated):
//   vis_in       : [n_ms, nrow, nchan_old, ncorr]  float2
//   flag_in      : [n_ms, nrow, nchan_old, ncorr]  uint8
//   uvw_in       : [n_ms, nrow, 3]                 double
//   freq_old_all : [n_ms, nchan_old]               double
//   freq_new_all : [n_ms, nchan_new]               double
//   df_old_k_all : [n_ms, nchan_old]               double  (FIX B)
//   geom_all     : [n_ms, 14] double: [inv_z, dl, dm, dn, Rot[0..8], unused]
//   df_new_all   : [n_ms]              double
//   equispaced_all: [n_ms]             int
//   df_old_eq_all: [n_ms]              double
//   vis_shifted  : [n_ms*nrow, nchan_old, ncorr]  cuDoubleComplex (scratch)
// ===========================================================================
__global__ void full_pipeline_batch_kernel(
    const float2*          __restrict__ vis_in,
    const uint8_t*         __restrict__ flag_in,
    const double*          __restrict__ uvw_in,
    const double*          __restrict__ freq_old_all,
    const double*          __restrict__ freq_new_all,
    const double*          __restrict__ df_old_k_all,
    const double*          __restrict__ geom_all,
    const double*          __restrict__ df_new_all,
    const int*             __restrict__ equispaced_all,
    const double*          __restrict__ df_old_eq_all,
    int n_ms, int nrow, int nchan_old, int nchan_new, int ncorr,
    float2*  __restrict__ vis_out,
    uint8_t* __restrict__ flag_out,
    double*  __restrict__ uvw_out,
    cuDoubleComplex* __restrict__ vis_shifted  // scratch [n_ms*nrow, nchan_old, ncorr]
)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int total  = n_ms * nrow;
    if (tid >= total) return;

    int ms_idx = tid / nrow;  // which MS this thread belongs to
    int i      = tid % nrow;  // baseline row index within that MS

    // Pointer arithmetic to locate this MS's slice within the flat buffers
    size_t vis_in_off   = (size_t)ms_idx * nrow * nchan_old * ncorr;
    size_t uvw_in_off   = (size_t)ms_idx * nrow * 3;
    size_t freq_old_off = (size_t)ms_idx * nchan_old;
    size_t freq_new_off = (size_t)ms_idx * nchan_new;
    size_t vis_sh_off   = (size_t)tid * nchan_old * ncorr;   // unique per (ms, row)
    size_t vis_out_off  = (size_t)ms_idx * nrow * nchan_new * ncorr;
    size_t uvw_out_off  = (size_t)ms_idx * nrow * 3;

    const float2*  vin_ms  = vis_in  + vis_in_off;
    const uint8_t* fin_ms  = flag_in + vis_in_off;
    const double*  uin_ms  = uvw_in  + uvw_in_off;
    const double*  fo      = freq_old_all + freq_old_off;
    const double*  fn      = freq_new_all + freq_new_off;
    const double*  dfk     = df_old_k_all + freq_old_off;
    float2*        vout_ms = vis_out  + vis_out_off;
    uint8_t*       fout_ms = flag_out + vis_out_off;
    double*        uout_ms = uvw_out  + uvw_out_off;
    cuDoubleComplex* vis_sh = vis_shifted + vis_sh_off;

    // Read geometric parameters for this MS from the packed geom_all array
    const double* g = geom_all + ms_idx * 14;
    double inv_z   = g[0];
    double dl      = g[1], dm = g[2], dn = g[3];
    double Rot0=g[4],  Rot1=g[5],  Rot2=g[6];
    double Rot3=g[7],  Rot4=g[8],  Rot5=g[9];
    double Rot6=g[10], Rot7=g[11], Rot8=g[12];

    double df_new    = df_new_all[ms_idx];
    int    equisp    = equispaced_all[ms_idx];
    double df_old_eq = df_old_eq_all[ms_idx];

    // --- Step 1: rest-frame UVW and rotate to new phase centre ---
    double u_rf = uin_ms[i*3+0] * inv_z;
    double v_rf = uin_ms[i*3+1] * inv_z;
    double w_rf = uin_ms[i*3+2] * inv_z;
    uout_ms[i*3+0] = Rot0*u_rf + Rot1*v_rf + Rot2*w_rf;
    uout_ms[i*3+1] = Rot3*u_rf + Rot4*v_rf + Rot5*w_rf;
    uout_ms[i*3+2] = Rot6*u_rf + Rot7*v_rf + Rot8*w_rf;
    double proj = u_rf*dl + v_rf*dm + w_rf*dn;

    // --- Step 2: phase shift (FIX A: inline float32 -> float64 cast) ---
    if (equisp && nchan_old > 1) {
        double phi0 = -TWO_PI * proj * fo[0] / C_LIGHT;
        double dphi = -TWO_PI * proj * df_old_eq / C_LIGHT;
        for (int j = 0; j < nchan_old; ++j) {
            double phi = phi0 + (double)j * dphi;
            double cos_phi, sin_phi;
            sincos(phi, &sin_phi, &cos_phi);
            for (int c = 0; c < ncorr; ++c) {
                float2 vf = vin_ms[(i*nchan_old+j)*ncorr+c];
                double vr = (double)vf.x, vi = (double)vf.y;
                vis_sh[j*ncorr+c] = make_cuDoubleComplex(
                    vr*cos_phi - vi*sin_phi,
                    vr*sin_phi + vi*cos_phi);
            }
        }
    } else {
        for (int j = 0; j < nchan_old; ++j) {
            double phi = -TWO_PI * proj * fo[j] / C_LIGHT;
            double cos_phi, sin_phi;
            sincos(phi, &sin_phi, &cos_phi);
            for (int c = 0; c < ncorr; ++c) {
                float2 vf = vin_ms[(i*nchan_old+j)*ncorr+c];
                double vr = (double)vf.x, vi = (double)vf.y;
                vis_sh[j*ncorr+c] = make_cuDoubleComplex(
                    vr*cos_phi - vi*sin_phi,
                    vr*sin_phi + vi*cos_phi);
            }
        }
    }

    // --- Step 3: overlap-weighted rebinning (FIX B: per-channel width) ---
    double df_approx = (nchan_old > 1) ? fabs(fo[1]-fo[0]) : 1.0;
    for (int c = 0; c < ncorr; ++c) {
        for (int j = 0; j < nchan_new; ++j) {
            double accum_r=0.0, accum_i=0.0, out_wgt=0.0;
            double lo_out = fn[j] - df_new*0.5;
            double hi_out = fn[j] + df_new*0.5;
            int k0 = max(0,           (int)floor((lo_out-fo[0]-df_approx*0.5)/df_approx)-1);
            int k1 = min(nchan_old-1, (int)floor((hi_out-fo[0]+df_approx*0.5)/df_approx)+1);
            for (int k = k0; k <= k1; ++k) {
                if (fin_ms[(i*nchan_old+k)*ncorr+c]) continue;
                double dfo   = dfk[k];
                double lo_in = fo[k] - dfo*0.5;
                double hi_in = fo[k] + dfo*0.5;
                double ov = fmax(0.0, fmin(hi_in,hi_out)-fmax(lo_in,lo_out));
                double w  = ov/dfo;
                if (w <= 0.0) continue;
                cuDoubleComplex vs = vis_sh[k*ncorr+c];
                accum_r += cuCreal(vs)*w;
                accum_i += cuCimag(vs)*w;
                out_wgt += w;
            }
            int oidx = (i*nchan_new+j)*ncorr+c;
            if (out_wgt > 0.0) {
                vout_ms[oidx] = make_float2((float)(accum_r/out_wgt), (float)(accum_i/out_wgt));
                fout_ms[oidx] = 0;
            } else {
                vout_ms[oidx] = make_float2(0.0f, 0.0f);
                fout_ms[oidx] = 1;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// full_pipeline_batch_cuda
// ---------------------------------------------------------------------------
// Batch GPU entry point.  Processes n_ms Measurement Sets in a single kernel
// launch.  Called from ms_ops.cpp::full_pipeline_batch() when _gpu_available
// is true.
//
// All input MS must have the same (nrow, nchan_old, ncorr) dimensions; the
// pipeline guarantees this because each chunk is processed as a single SPW.
//
// Workflow:
//   1. Compute geometric parameters and per-channel widths for all MSs on
//      the host and pack them into flat arrays.
//   2. Allocate GPU buffers, copy all inputs in one batch.
//   3. Launch full_pipeline_batch_kernel (one thread per (MS, row)).
//   4. Copy results back and free GPU memory.
// ---------------------------------------------------------------------------
void full_pipeline_batch_cuda(
    const float*   vis_in_h,       // [n_ms, nrow, nchan_old, ncorr] complex64
    const uint8_t* flag_in_h,      // [n_ms, nrow, nchan_old, ncorr]
    const double*  uvw_in_h,       // [n_ms, nrow, 3]
    const double*  freq_old_h,     // [n_ms, nchan_old]
    const double*  freq_new_h,     // [n_ms, nchan_new]
    const double*  z_h,            // [n_ms]
    const double*  ra_old_h,       // [n_ms]
    const double*  dec_old_h,      // [n_ms]
    const double*  ra_new_h,       // [n_ms]
    const double*  dec_new_h,      // [n_ms]
    int n_ms, int nrow, int nchan_old, int nchan_new, int ncorr,
    float*   vis_out_h,            // [n_ms, nrow, nchan_new, ncorr] complex64
    uint8_t* flag_out_h,           // [n_ms, nrow, nchan_new, ncorr]
    double*  uvw_out_h             // [n_ms, nrow, 3]
)
{
    // --- Compute per-MS geometric parameters on the host ---
    // geom_all[m] = [inv_z, dl, dm, dn, Rot[0..8], 0]  (14 doubles per MS)
    std::vector<double> geom_h(n_ms * 14);
    std::vector<double> df_old_k_h(n_ms * nchan_old);
    std::vector<double> df_new_h(n_ms);
    std::vector<int>    equispaced_h(n_ms);
    std::vector<double> df_old_eq_h(n_ms);

    for (int m = 0; m < n_ms; ++m) {
        const double* fo = freq_old_h + m * nchan_old;
        const double* fn = freq_new_h + m * nchan_new;
        double z       = z_h[m];
        double ra_old  = ra_old_h[m], dec_old = dec_old_h[m];
        double ra_new  = ra_new_h[m], dec_new = dec_new_h[m];

        // Rotation matrix
        double basis_old[9], basis_new[9], Rot[9];
        build_basis_host(ra_old, dec_old, basis_old);
        build_basis_host(ra_new, dec_new, basis_new);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                Rot[i*3+j] = 0.0;
                for (int k = 0; k < 3; ++k)
                    Rot[i*3+j] += basis_new[i*3+k] * basis_old[j*3+k];
            }
        double dl   = (ra_new-ra_old)*cos(dec_old);
        double dm_  = dec_new-dec_old;
        double arg  = 1.0-dl*dl-dm_*dm_;
        double dn   = (arg>0.0?sqrt(arg):0.0)-1.0;
        double inv_z = 1.0/(1.0+z);

        double* g = geom_h.data() + m*14;
        g[0]=inv_z; g[1]=dl;    g[2]=dm_;   g[3]=dn;
        g[4]=Rot[0]; g[5]=Rot[1]; g[6]=Rot[2];
        g[7]=Rot[3]; g[8]=Rot[4]; g[9]=Rot[5];
        g[10]=Rot[6]; g[11]=Rot[7]; g[12]=Rot[8];
        g[13]=0.0;

        // Per-channel input widths (FIX B)
        double* dfk = df_old_k_h.data() + m*nchan_old;
        if (nchan_old == 1) {
            dfk[0] = (nchan_new>1) ? fabs(fn[1]-fn[0]) : 1.0;
        } else {
            dfk[0] = fo[1]-fo[0];
            for (int k = 1; k < nchan_old-1; ++k)
                dfk[k] = (fo[k+1]-fo[k-1])*0.5;
            dfk[nchan_old-1] = fo[nchan_old-1]-fo[nchan_old-2];
        }
        for (int k = 0; k < nchan_old; ++k) dfk[k] = fabs(dfk[k]);

        df_new_h[m] = (nchan_new>1) ? fabs(fn[1]-fn[0]) : 1.0;

        int eq=1;
        double dfeq = (nchan_old>1) ? fo[1]-fo[0] : 0.0;
        for (int j=2; j<nchan_old&&eq; ++j)
            if (fabs((fo[j]-fo[j-1])-dfeq)>fabs(dfeq)*1e-6) eq=0;
        equispaced_h[m] = eq;
        df_old_eq_h[m]  = dfeq;
    }

    // --- GPU buffer sizes ---
    size_t sz_vis_in   = (size_t)n_ms*nrow*nchan_old*ncorr*sizeof(float2);
    size_t sz_flag_in  = (size_t)n_ms*nrow*nchan_old*ncorr;
    size_t sz_uvw_in   = (size_t)n_ms*nrow*3*sizeof(double);
    size_t sz_fo       = (size_t)n_ms*nchan_old*sizeof(double);
    size_t sz_fn       = (size_t)n_ms*nchan_new*sizeof(double);
    size_t sz_dfk      = (size_t)n_ms*nchan_old*sizeof(double);
    size_t sz_geom     = (size_t)n_ms*14*sizeof(double);
    size_t sz_dfnew    = (size_t)n_ms*sizeof(double);
    size_t sz_eq       = (size_t)n_ms*sizeof(int);
    size_t sz_dfeq     = (size_t)n_ms*sizeof(double);
    size_t sz_vis_out  = (size_t)n_ms*nrow*nchan_new*ncorr*sizeof(float2);
    size_t sz_flag_out = (size_t)n_ms*nrow*nchan_new*ncorr;
    size_t sz_uvw_out  = (size_t)n_ms*nrow*3*sizeof(double);
    size_t sz_vis_sh   = (size_t)n_ms*nrow*nchan_old*ncorr*sizeof(cuDoubleComplex);

    // --- Allocate GPU buffers ---
    float2 *d_vis_in, *d_vis_out;
    uint8_t *d_flag_in, *d_flag_out;
    double *d_uvw_in, *d_fo, *d_fn, *d_dfk, *d_geom, *d_dfnew, *d_dfeq, *d_uvw_out;
    int *d_eq;
    cuDoubleComplex *d_vis_sh;

    CUDA_CHECK(cudaMalloc(&d_vis_in,  sz_vis_in));
    CUDA_CHECK(cudaMalloc(&d_flag_in, sz_flag_in));
    CUDA_CHECK(cudaMalloc(&d_uvw_in,  sz_uvw_in));
    CUDA_CHECK(cudaMalloc(&d_fo,      sz_fo));
    CUDA_CHECK(cudaMalloc(&d_fn,      sz_fn));
    CUDA_CHECK(cudaMalloc(&d_dfk,     sz_dfk));
    CUDA_CHECK(cudaMalloc(&d_geom,    sz_geom));
    CUDA_CHECK(cudaMalloc(&d_dfnew,   sz_dfnew));
    CUDA_CHECK(cudaMalloc(&d_eq,      sz_eq));
    CUDA_CHECK(cudaMalloc(&d_dfeq,    sz_dfeq));
    CUDA_CHECK(cudaMalloc(&d_vis_out, sz_vis_out));
    CUDA_CHECK(cudaMalloc(&d_flag_out,sz_flag_out));
    CUDA_CHECK(cudaMalloc(&d_uvw_out, sz_uvw_out));
    CUDA_CHECK(cudaMalloc(&d_vis_sh,  sz_vis_sh));

    // --- Copy inputs host -> device ---
    CUDA_CHECK(cudaMemcpy(d_vis_in,  vis_in_h,          sz_vis_in,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flag_in, flag_in_h,         sz_flag_in, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uvw_in,  uvw_in_h,          sz_uvw_in,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fo,      freq_old_h,        sz_fo,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fn,      freq_new_h,        sz_fn,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dfk,     df_old_k_h.data(), sz_dfk,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_geom,    geom_h.data(),     sz_geom,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dfnew,   df_new_h.data(),   sz_dfnew,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eq,      equispaced_h.data(),sz_eq,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dfeq,    df_old_eq_h.data(),sz_dfeq,    cudaMemcpyHostToDevice));

    // --- Kernel launch: one thread per (MS, row) ---
    int total_threads = n_ms * nrow;
    int threads = 128;
    int blocks  = (total_threads + threads - 1) / threads;

    full_pipeline_batch_kernel<<<blocks, threads>>>(
        d_vis_in, d_flag_in, d_uvw_in,
        d_fo, d_fn, d_dfk, d_geom, d_dfnew, d_eq, d_dfeq,
        n_ms, nrow, nchan_old, nchan_new, ncorr,
        d_vis_out, d_flag_out, d_uvw_out, d_vis_sh
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy results device -> host ---
    CUDA_CHECK(cudaMemcpy(vis_out_h,  d_vis_out,  sz_vis_out,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flag_out_h, d_flag_out, sz_flag_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uvw_out_h,  d_uvw_out,  sz_uvw_out,  cudaMemcpyDeviceToHost));

    // --- Free GPU buffers ---
    cudaFree(d_vis_in);  cudaFree(d_flag_in);  cudaFree(d_uvw_in);
    cudaFree(d_fo);      cudaFree(d_fn);        cudaFree(d_dfk);
    cudaFree(d_geom);    cudaFree(d_dfnew);     cudaFree(d_eq);
    cudaFree(d_dfeq);    cudaFree(d_vis_out);   cudaFree(d_flag_out);
    cudaFree(d_uvw_out); cudaFree(d_vis_sh);
}

}  // extern "C"
