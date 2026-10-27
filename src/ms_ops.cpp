// =============================================================================
// ms_ops.cpp — compute kernel of ViSta (Python module "ms_ops")
// =============================================================================
//
// For every row of a chunk of visibilities:
//   1. rest-frame UVW:   u_rf = u_obs / (1 + z)
//   2. rotation of UVW towards the new phase centre
//   3. phase shift:      V * exp(-2 pi i nu_rf (w_new - w_old) / c),
//                        exact: w_new - w_old = u l + v m + w (n - 1),
//                        computed in float64 (the input is complex64)
//   4. rebinning onto the output grid, weighted by the channel overlap, with
//      the per-channel input width (centred difference) as CASA mstransform
//
// CPU   full_pipeline()            OpenMP over the rows; the rebinning plan is
//                                  built once per call and the work buffers
//                                  once per thread.  The GIL is released, so
//                                  the reader and writer of the pipeline keep
//                                  working while the kernel runs.
// GPU   full_pipeline_batch_into() several chunks (of any size, from different
//                                  MSs) in one launch, written into output
//                                  buffers owned by the caller (reused and
//                                  page-locked with pin_host()).  Built only
//                                  with -DWITH_CUDA; without a GPU, or after a
//                                  GPU error, the same call runs on the CPU.
//
// The weight scale R = df_new / df_old_rf is applied in Python, not here.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <omp.h>

#include "ms_ops_cuda.h"      // VistaItem (+ the GPU C API in CUDA builds)

namespace py = pybind11;

// =============================================================================
// CPU kernel
// =============================================================================
namespace vista {

static constexpr double C_LIGHT = 299792458.0;
static constexpr double TWO_PI  = 2.0 * M_PI;

inline double median_df(const double* freq, ptrdiff_t n)
{
    if (n < 2) return 1.0;
    std::vector<double> diffs(n - 1);
    for (ptrdiff_t i = 0; i < n - 1; ++i)
        diffs[i] = std::abs(freq[i+1] - freq[i]);
    std::nth_element(diffs.begin(), diffs.begin() + diffs.size()/2, diffs.end());
    return diffs[diffs.size()/2];
}

inline void build_basis(double ra, double dec, double M[9])
{
    double sr = std::sin(ra),  cr = std::cos(ra);
    double sd = std::sin(dec), cd = std::cos(dec);
    M[0] = -sr;    M[1] =  cr;    M[2] = 0.0;
    M[3] = -cr*sd; M[4] = -sr*sd; M[5] = cd;
    M[6] =  cr*cd; M[7] =  sr*cd; M[8] = sd;
}

inline void mat_mul_BT(const double A[9], const double B[9], double R[9])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R[i*3+j] = 0.0;
            for (int k = 0; k < 3; ++k)
                R[i*3+j] += A[i*3+k] * B[j*3+k];
        }
}

// All arrays are C-ordered: vin/fin [nrow, nchan_old, ncorr], uin [nrow, 3],
// vout/fout [nrow, nchan_new, ncorr], uout [nrow, 3].
inline void cpu_full_pipeline(
    const std::complex<float>* vin, const bool* fin, const double* uin,
    ptrdiff_t nrow, ptrdiff_t nchan_old, ptrdiff_t ncorr,
    double z, const double* freq_old_rf,
    double ra_old, double dec_old, double ra_new, double dec_new,
    const double* freq_new, ptrdiff_t nchan_new,
    std::complex<float>* vout, bool* fout, double* uout,
    const double* width_old = nullptr)
{
    typedef ptrdiff_t ssize_t;

    // ---- per-channel width of the input grid (CHAN_WIDTH when given) ----
    std::vector<double> df_old_k;
    vista_df_old_k(freq_old_rf, nchan_old, freq_new, nchan_new, df_old_k, width_old);
    const double df_new = median_df(freq_new, nchan_new);

    // equally spaced input grid -> phase by recurrence
    bool eq_old = (nchan_old > 1);
    const double df_old_eq = (nchan_old > 1) ? (freq_old_rf[1] - freq_old_rf[0]) : 0.0;
    for (ssize_t j = 2; j < nchan_old && eq_old; ++j)
        if (std::abs((freq_old_rf[j]-freq_old_rf[j-1])-df_old_eq) > std::abs(df_old_eq)*1e-6)
            eq_old = false;

    double basis_old[9], basis_new_m[9], Rot[9];
    build_basis(ra_old, dec_old, basis_old);
    build_basis(ra_new, dec_new, basis_new_m);
    mat_mul_BT(basis_new_m, basis_old, Rot);
    const double inv_z = 1.0 / (1.0 + z);

    // ---- rebinning plan, once per call ----
    // For every input channel k: the list of (j, w) with w > 0, j increasing.
    // Same formulas and same order as a per-row loop -> same bits.
    std::vector<int64_t> plan_off(nchan_old + 1, 0);
    std::vector<int64_t> plan_j;
    std::vector<double>  plan_w;
    plan_j.reserve(nchan_old * 3);
    plan_w.reserve(nchan_old * 3);
    for (ssize_t k = 0; k < nchan_old; ++k) {
        const double dfo   = df_old_k[k];
        const double lo_in = freq_old_rf[k] - dfo * 0.5;
        const double hi_in = freq_old_rf[k] + dfo * 0.5;
        const ssize_t j0 = std::max((ssize_t)0,
            (ssize_t)std::floor((lo_in - (freq_new[0] + df_new * 0.5)) / df_new));
        const ssize_t j1 = std::min(nchan_new - 1,
            (ssize_t)std::floor((hi_in - (freq_new[0] - df_new * 0.5)) / df_new));
        for (ssize_t j = j0; j <= j1; ++j) {
            const double lo_out = freq_new[j] - df_new * 0.5;
            const double hi_out = freq_new[j] + df_new * 0.5;
            const double ov = std::max(0.0,
                std::min(hi_in, hi_out) - std::max(lo_in, lo_out));
            const double w = ov / dfo;
            if (w <= 0.0) continue;
            plan_j.push_back(j);
            plan_w.push_back(w);
        }
        plan_off[k + 1] = (int64_t)plan_j.size();
    }
    const int64_t* pj  = plan_j.data();
    const double*  pw  = plan_w.data();
    const int64_t* off = plan_off.data();

    #pragma omp parallel
    {
        std::vector<std::complex<double>> vis_shifted(nchan_old * ncorr);
        std::vector<std::complex<double>> accum(nchan_new);
        std::vector<double>               out_wgt(nchan_new);

        #pragma omp for schedule(dynamic, 64)
        for (ssize_t i = 0; i < nrow; ++i) {
            // 1-2. rest-frame UVW and rotation
            const double u_rf = uin[i*3+0] * inv_z;
            const double v_rf = uin[i*3+1] * inv_z;
            const double w_rf = uin[i*3+2] * inv_z;
            uout[i*3+0] = Rot[0]*u_rf + Rot[1]*v_rf + Rot[2]*w_rf;
            uout[i*3+1] = Rot[3]*u_rf + Rot[4]*v_rf + Rot[5]*w_rf;
            uout[i*3+2] = Rot[6]*u_rf + Rot[7]*v_rf + Rot[8]*w_rf;
            // exact delay difference between the old and the new phase centre:
            //   u l + v m + w (n - 1) = w_new - w_old
            // (l, m, n = direction cosines of the new centre in the old frame;
            //  the former small-angle dl, dm gave an error ~ offset^2 tan(dec))
            const double proj = uout[i*3+2] - w_rf;

            // 3. phase shift in float64
            const std::complex<float>* vrow = vin + (size_t)i * nchan_old * ncorr;
            if (eq_old && nchan_old > 1) {
                const double phi0 = -TWO_PI * proj * freq_old_rf[0] / C_LIGHT;
                const double dphi = -TWO_PI * proj * df_old_eq      / C_LIGHT;
                std::complex<double> ph   = std::polar(1.0, phi0);
                const std::complex<double> step = std::polar(1.0, dphi);
                for (ssize_t j = 0; j < nchan_old; ++j) {
                    if (j % 64 == 0) ph = std::polar(1.0, phi0 + j * dphi);
                    for (ssize_t c = 0; c < ncorr; ++c)
                        vis_shifted[j*ncorr+c] =
                            static_cast<std::complex<double>>(vrow[j*ncorr+c]) * ph;
                    ph *= step;
                }
            } else {
                for (ssize_t j = 0; j < nchan_old; ++j) {
                    const double phi = -TWO_PI * proj * freq_old_rf[j] / C_LIGHT;
                    const std::complex<double> ph = std::polar(1.0, phi);
                    for (ssize_t c = 0; c < ncorr; ++c)
                        vis_shifted[j*ncorr+c] =
                            static_cast<std::complex<double>>(vrow[j*ncorr+c]) * ph;
                }
            }

            // 4. overlap-weighted rebinning; output channels with no data are flagged
            const bool* frow = fin + (size_t)i * nchan_old * ncorr;
            std::complex<float>* vo = vout + (size_t)i * nchan_new * ncorr;
            bool*                fo = fout + (size_t)i * nchan_new * ncorr;
            for (ssize_t c = 0; c < ncorr; ++c) {
                std::fill(accum.begin(),   accum.end(),   std::complex<double>(0.0, 0.0));
                std::fill(out_wgt.begin(), out_wgt.end(), 0.0);
                for (ssize_t k = 0; k < nchan_old; ++k) {
                    if (frow[k*ncorr + c]) continue;
                    const std::complex<double> vk = vis_shifted[k*ncorr+c];
                    for (int64_t p = off[k]; p < off[k+1]; ++p) {
                        accum[pj[p]]   += vk * pw[p];
                        out_wgt[pj[p]] += pw[p];
                    }
                }
                for (ssize_t j = 0; j < nchan_new; ++j) {
                    if (out_wgt[j] > 0.0) {
                        vo[j*ncorr+c] = static_cast<std::complex<float>>(accum[j] / out_wgt[j]);
                        fo[j*ncorr+c] = false;
                    } else {
                        vo[j*ncorr+c] = {0.0f, 0.0f};
                        fo[j*ncorr+c] = true;
                    }
                }
            }
        }
    }
}

// The same computation on a VistaItem (CPU fallback of the batch call)
inline void cpu_run_item(const VistaItem& it)
{
    if (it.nrow == 0) return;
    cpu_full_pipeline(
        reinterpret_cast<const std::complex<float>*>(it.vis_in),
        reinterpret_cast<const bool*>(it.flag_in), it.uvw_in,
        it.nrow, it.nchan_old, it.ncorr, it.z, it.freq_old,
        it.ra_old, it.dec_old, it.ra_new, it.dec_new, it.freq_new, it.nchan_new,
        reinterpret_cast<std::complex<float>*>(it.vis_out),
        reinterpret_cast<bool*>(it.flag_out), it.uvw_out, it.width_old);
}

}  // namespace vista

// =============================================================================
// Python types
// =============================================================================
using VisArr  = py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>;
using FlagArr = py::array_t<bool,                py::array::c_style | py::array::forcecast>;
using DblArr  = py::array_t<double,              py::array::c_style | py::array::forcecast>;
using VisOut  = py::array_t<std::complex<float>, py::array::c_style>;
using FlagOut = py::array_t<bool,                py::array::c_style>;
using DblOut  = py::array_t<double,              py::array::c_style>;
using Result  = std::tuple<py::array_t<std::complex<float>>, py::array_t<bool>, py::array_t<double>>;

// =============================================================================
// full_pipeline: one chunk on the CPU
// =============================================================================
Result full_pipeline(VisArr vis, FlagArr flag, DblArr uvw, double z, DblArr freq_old_rf,
                     double ra_old, double dec_old, double ra_new, double dec_new,
                     DblArr freq_new, py::object width_old_rf)
{
    if (vis.ndim() != 3) throw std::invalid_argument("vis must be (nrow, nchan, ncorr)");
    const ssize_t nrow = vis.shape(0), nco = vis.shape(1), ncr = vis.shape(2);
    if (freq_old_rf.ndim() != 1 || freq_old_rf.shape(0) != nco)
        throw std::invalid_argument("freq_old_rf must have vis.shape[1] elements");
    if (freq_new.ndim() != 1 || freq_new.shape(0) < 1)
        throw std::invalid_argument("freq_new is empty");
    if (flag.ndim() != 3 || flag.shape(0) != nrow || flag.shape(1) != nco || flag.shape(2) != ncr)
        throw std::invalid_argument("flag must have the shape of vis");
    if (uvw.ndim() != 2 || uvw.shape(0) != nrow || uvw.shape(1) != 3)
        throw std::invalid_argument("uvw must be (nrow, 3)");
    const ssize_t ncn = freq_new.shape(0);
    DblArr width;
    const double* pw = nullptr;
    if (!width_old_rf.is_none()) {
        width = width_old_rf.cast<DblArr>();
        if (width.ndim() != 1 || width.shape(0) != nco)
            throw std::invalid_argument("width_old_rf must have vis.shape[1] elements");
        pw = width.data();
    }

    py::array_t<std::complex<float>> vo({nrow, ncn, ncr});
    py::array_t<bool>                fo({nrow, ncn, ncr});
    py::array_t<double>              uo({nrow, (ssize_t)3});
    const std::complex<float>* pv = vis.data();
    const bool* pf = flag.data();
    const double* pu = uvw.data();
    const double* pfo = freq_old_rf.data();
    const double* pfn = freq_new.data();
    std::complex<float>* ov = vo.mutable_data();
    bool* of = fo.mutable_data();
    double* ou = uo.mutable_data();
    {
        py::gil_scoped_release nogil;
        vista::cpu_full_pipeline(pv, pf, pu, nrow, nco, ncr, z, pfo,
                                 ra_old, dec_old, ra_new, dec_new, pfn, ncn, ov, of, ou, pw);
    }
    return {vo, fo, uo};
}

// =============================================================================
// GPU state
// =============================================================================
// -1 = not checked yet, 0 = off, 1 = on.  The check runs on the first request,
// not at import, so importing the module does not initialise CUDA.  After a
// GPU error the GPU is switched off for the rest of the process and the work
// continues on the CPU; the message is kept for gpu_last_error().
static std::atomic<int> g_gpu_state{-1};
static std::string      g_gpu_error;
static constexpr double GPU_MEM_FRACTION = 0.85;   // share of the free GPU memory a call may use

static bool gpu_enabled()
{
#ifdef WITH_CUDA
    int s = g_gpu_state.load();
    if (s < 0) {
        s = vista_cuda_device_count() > 0 ? 1 : 0;
        g_gpu_state.store(s);
    }
    return s == 1;
#else
    return false;
#endif
}

// Runs a list of blocks: GPU when available, otherwise (or after a GPU error)
// CPU.  Called without the GIL.
static void run_items_nogil(const std::vector<VistaItem>& items)
{
#ifdef WITH_CUDA
    if (gpu_enabled()) {
        char err[1024] = {0};
        if (vista_cuda_run(items.data(), (int)items.size(), GPU_MEM_FRACTION,
                           err, sizeof(err)) == 0)
            return;
        g_gpu_error = err;
        g_gpu_state.store(0);
    }
#endif
    for (const VistaItem& it : items) vista::cpu_run_item(it);
}

// =============================================================================
// full_pipeline_batch_into: N chunks in one call, into caller-owned outputs
// =============================================================================
// Inputs are Python lists of the same length; the chunks may have different
// shapes.  The outputs must be C-contiguous, of the right dtype and shaped
// [nrow, nchan_new, ncorr] / [nrow, 3].  On the GPU the work is packed and,
// if it does not fit in the free memory, split into passes.
void full_pipeline_batch_into(
    py::list vis_list, py::list flag_list, py::list uvw_list,
    py::list z_list, py::list freq_old_list,
    py::list ra_old_list, py::list dec_old_list,
    py::list ra_new_list, py::list dec_new_list,
    py::list freq_new_list,
    py::list vis_out_list, py::list flag_out_list, py::list uvw_out_list,
    py::object width_old_list)
{
    const bool has_w = !width_old_list.is_none();
    const size_t n = py::len(vis_list);
    for (const py::list& l : {flag_list, uvw_list, z_list, freq_old_list, ra_old_list,
                              dec_old_list, ra_new_list, dec_new_list, freq_new_list,
                              vis_out_list, flag_out_list, uvw_out_list})
        if (py::len(l) != n)
            throw std::invalid_argument("full_pipeline_batch_into: the lists must have the same length");

    struct Arrays { VisArr vis; FlagArr flag; DblArr uvw; DblArr fo; DblArr fn;
                    VisOut vo; FlagOut flo; DblOut uo; DblArr w; };
    std::vector<Arrays>    arrays(n);       // keep inputs and outputs alive
    std::vector<VistaItem> items(n);
    for (size_t m = 0; m < n; ++m) {
        Arrays& a = arrays[m];
        const std::string tag = "chunk " + std::to_string(m) + ": ";
        a.vis  = vis_list[m].cast<VisArr>();
        a.flag = flag_list[m].cast<FlagArr>();
        a.uvw  = uvw_list[m].cast<DblArr>();
        a.fo   = freq_old_list[m].cast<DblArr>();
        a.fn   = freq_new_list[m].cast<DblArr>();
        a.vo   = vis_out_list[m].cast<VisOut>();
        a.flo  = flag_out_list[m].cast<FlagOut>();
        a.uo   = uvw_out_list[m].cast<DblOut>();
        if (a.vis.ndim() != 3) throw std::invalid_argument(tag + "vis must be (nrow, nchan, ncorr)");
        const ssize_t nrow = a.vis.shape(0), nco = a.vis.shape(1), ncr = a.vis.shape(2);
        const ssize_t ncn = a.fn.ndim() == 1 ? a.fn.shape(0) : -1;
        if (a.flag.ndim() != 3 || a.flag.shape(0) != nrow || a.flag.shape(1) != nco || a.flag.shape(2) != ncr)
            throw std::invalid_argument(tag + "flag must have the shape of vis");
        if (a.uvw.ndim() != 2 || a.uvw.shape(0) != nrow || a.uvw.shape(1) != 3)
            throw std::invalid_argument(tag + "uvw must be (nrow, 3)");
        if (a.fo.ndim() != 1 || a.fo.shape(0) != nco || ncn < 1 || nco < 1 || ncr < 1
            || ncr > VISTA_MAX_CORR)
            throw std::invalid_argument(tag + "invalid frequency grids or correlations");
        if (a.vo.ndim() != 3 || a.vo.shape(0) != nrow || a.vo.shape(1) != ncn || a.vo.shape(2) != ncr
            || a.flo.ndim() != 3 || a.flo.shape(0) != nrow || a.flo.shape(1) != ncn || a.flo.shape(2) != ncr
            || a.uo.ndim() != 2 || a.uo.shape(0) != nrow || a.uo.shape(1) != 3)
            throw std::invalid_argument(tag + "output arrays have the wrong shape");
        VistaItem& it = items[m];
        it.vis_in   = reinterpret_cast<const float*>(a.vis.data());
        it.flag_in  = reinterpret_cast<const uint8_t*>(a.flag.data());
        it.uvw_in   = a.uvw.data();
        it.freq_old = a.fo.data();
        it.freq_new = a.fn.data();
        it.nrow = nrow; it.nchan_old = (int32_t)nco; it.nchan_new = (int32_t)ncn; it.ncorr = (int32_t)ncr;
        it.z = z_list[m].cast<double>();
        it.ra_old = ra_old_list[m].cast<double>(); it.dec_old = dec_old_list[m].cast<double>();
        it.ra_new = ra_new_list[m].cast<double>(); it.dec_new = dec_new_list[m].cast<double>();
        it.vis_out  = reinterpret_cast<float*>(a.vo.mutable_data());
        it.flag_out = reinterpret_cast<uint8_t*>(a.flo.mutable_data());
        it.uvw_out  = a.uo.mutable_data();
        if (has_w) {
            a.w = py::list(width_old_list)[m].cast<DblArr>();
            if (a.w.ndim() != 1 || a.w.shape(0) != nco)
                throw std::invalid_argument(tag + "width_old must have one value per input channel");
            it.width_old = a.w.data();
        }
    }
    py::gil_scoped_release nogil;
    run_items_nogil(items);
}

// =============================================================================
// Page-locked host memory (the reused output buffers of the GPU pipeline)
// =============================================================================
static bool py_pin_host(py::array a)
{
#ifdef WITH_CUDA
    if (!gpu_enabled()) return false;
    return vista_cuda_host_register(a.mutable_data(), (size_t)a.nbytes()) == 0;
#else
    (void)a; return false;
#endif
}

static bool py_unpin_host(py::array a)
{
#ifdef WITH_CUDA
    return vista_cuda_host_unregister(a.mutable_data()) == 0;
#else
    (void)a; return false;
#endif
}

static std::pair<size_t, size_t> py_gpu_memory_info()
{
#ifdef WITH_CUDA
    if (gpu_enabled()) {
        size_t f = 0, t = 0;
        if (vista_cuda_mem_info(&f, &t) == 0) return {f, t};
    }
#endif
    return {0, 0};
}

static bool py_built_with_cuda()
{
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

// =============================================================================
// Python module
// =============================================================================
PYBIND11_MODULE(ms_ops, m)
{
    m.doc() = "ViSta compute kernel: rest-framing, phase shift and rebinning "
              "(CPU OpenMP, optional CUDA)";

    m.def("full_pipeline", &full_pipeline,
          py::arg("vis"), py::arg("flag"), py::arg("uvw"), py::arg("z"), py::arg("freq_old_rf"),
          py::arg("ra_old"), py::arg("dec_old"), py::arg("ra_new"), py::arg("dec_new"),
          py::arg("freq_new"), py::arg("width_old_rf") = py::none(),
          "One chunk on the CPU (OpenMP, GIL released).  "
          "Returns (vis_out, flag_out, uvw_out).");

    m.def("full_pipeline_batch_into", &full_pipeline_batch_into,
          py::arg("vis_list"), py::arg("flag_list"), py::arg("uvw_list"),
          py::arg("z_list"), py::arg("freq_old_list"),
          py::arg("ra_old_list"), py::arg("dec_old_list"),
          py::arg("ra_new_list"), py::arg("dec_new_list"),
          py::arg("freq_new_list"),
          py::arg("vis_out_list"), py::arg("flag_out_list"), py::arg("uvw_out_list"),
          py::arg("width_old_list") = py::none(),
          "Several chunks (of any shape) in one call, written into the given "
          "output arrays: one GPU launch, split into passes if needed; on the "
          "CPU without a GPU or after a GPU error.");

    m.def("pin_host", &py_pin_host, py::arg("array"),
          "Page-lock a numpy array for fast GPU copies (cudaHostRegister); True on success.");
    m.def("unpin_host", &py_unpin_host, py::arg("array"));

    m.def("gpu_available", []() { return gpu_enabled(); },
          "True if the GPU will be used: CUDA build, a visible GPU and no GPU error so far.");
    m.def("gpu_last_error", []() { return g_gpu_error; },
          "Message of the GPU error that switched the GPU off ('' if none).");
    m.def("gpu_memory_info", &py_gpu_memory_info,
          "(free, total) GPU memory in bytes; (0, 0) without a GPU.");
    m.def("built_with_cuda", &py_built_with_cuda,
          "True if the module was compiled with CUDA support.");

    m.def("omp_max_threads", []() { return omp_get_max_threads(); },
          "OpenMP threads used by the CPU kernel.");
    m.def("omp_set_threads", [](int n) { omp_set_num_threads(n); }, py::arg("n"),
          "Set the OpenMP threads of the CPU kernel.");
}
