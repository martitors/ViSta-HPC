// The compute kernel called from C++, to compare with the same kernel called
// from Python (bench_binding.py): what the pybind11 interface costs.
//
//     g++ -O3 -fopenmp -std=c++17 $(python3 -m pybind11 --includes) \
//         bench_binding.cpp -o bench_binding $(python3-config --ldflags --embed)
//
// Three times are measured for every chunk size:
//
//   kernel         cpu_full_pipeline() on buffers allocated once;
//   kernel+alloc   the same, with the output buffers allocated at every call,
//                  as the binding has to do;
//   (Python)       ms_ops.full_pipeline(), measured by bench_binding.py.
//
// The data are built by the same formulas on both sides, and both print the
// checksum of the result: the two must agree, or they are not doing the same
// work.  The output is one JSON object per line.
#include "../../src/ms_ops.cpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

// numpy gives 64-byte aligned buffers: the C++ side allocates the same way,
// or the two would not be comparing the same memory.
struct Free { void operator()(void* p) const { std::free(p); } };
template <class T>
std::unique_ptr<T[], Free> aligned(size_t n)
{
    size_t bytes = ((n * sizeof(T) + 63) / 64) * 64;
    return std::unique_ptr<T[], Free>(static_cast<T*>(std::aligned_alloc(64, bytes)));
}

const double Z = 2.5, RA_OLD = 1.0, DEC_OLD = 0.3;
const double RA_NEW = 1.0 + 1e-5, DEC_NEW = 0.3 + 1e-5;

double now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv)
{
    int nchan_old = 1000, nchan_new = 800, ncorr = 2, threads = 1;
    std::vector<long> rows = {1, 8, 64, 512, 5000};
    for (int i = 1; i < argc - 1; ++i) {
        if (!std::strcmp(argv[i], "--threads"))   threads = std::atoi(argv[i + 1]);
        if (!std::strcmp(argv[i], "--nchan"))     nchan_old = std::atoi(argv[i + 1]);
        if (!std::strcmp(argv[i], "--nchan-new")) nchan_new = std::atoi(argv[i + 1]);
    }
    omp_set_num_threads(threads);

    // the two frequency grids: a 1.875 GHz band around CO(3-2) at z = 2.5
    std::vector<double> freq_old(nchan_old), freq_new(nchan_new);
    for (int c = 0; c < nchan_old; ++c)
        freq_old[c] = (340e9 + 1.875e9 * c / (nchan_old - 1.0)) * (1.0 + Z);
    for (int c = 0; c < nchan_new; ++c)
        freq_new[c] = freq_old[0] + (freq_old[nchan_old - 1] - freq_old[0])
                                    * (0.05 + 0.90 * c / (nchan_new - 1.0));

    for (long nrow : rows) {
        const size_t nin = (size_t)nrow * nchan_old * ncorr;
        const size_t nout = (size_t)nrow * nchan_new * ncorr;
        auto vis = aligned<std::complex<float>>(nin);
        auto flag = aligned<unsigned char>(nin);
        auto uvw = aligned<double>(nrow * 3);
        for (long r = 0; r < nrow; ++r) {
            for (int c = 0; c < nchan_old; ++c)
                for (int p = 0; p < ncorr; ++p) {
                    const size_t i = ((size_t)r * nchan_old + c) * ncorr + p;
                    const double a = 0.001 * (r + 3.0 * c + 7.0 * p);
                    vis.get()[i] = std::complex<float>((float)std::sin(a), (float)std::cos(a));
                    flag.get()[i] = ((r + c + p) % 97 == 0);
                }
            uvw.get()[r * 3 + 0] = 100.0 + 0.5 * r;
            uvw.get()[r * 3 + 1] = -50.0 + 0.25 * r;
            uvw.get()[r * 3 + 2] = 10.0 + 0.1 * r;
        }

        auto vo = aligned<std::complex<float>>(nout);
        auto fo = aligned<unsigned char>(nout);
        auto uo = aligned<double>(nrow * 3);
        auto call = [&](std::complex<float>* ov, bool* of, double* ou) {
            vista::cpu_full_pipeline(vis.get(), (const bool*)flag.get(), uvw.get(),
                                     nrow, nchan_old, ncorr, Z, freq_old.data(),
                                     RA_OLD, DEC_OLD, RA_NEW, DEC_NEW,
                                     freq_new.data(), nchan_new, ov, of, ou, nullptr);
        };

        const int reps = (int)std::max(3.0, std::min(200.0, 2e8 / (double)(nin + 1)));
        call(vo.get(), (bool*)fo.get(), uo.get());                         // warm-up
        std::vector<double> t_kernel, t_alloc;
        for (int k = 0; k < reps; ++k) {
            const double t0 = now();
            call(vo.get(), (bool*)fo.get(), uo.get());
            t_kernel.push_back(now() - t0);
        }
        for (int k = 0; k < reps; ++k) {
            const double t0 = now();
            auto v2 = aligned<std::complex<float>>(nout);
            auto f2 = aligned<unsigned char>(nout);
            auto u2 = aligned<double>(nrow * 3);
            call(v2.get(), (bool*)f2.get(), u2.get());
            t_alloc.push_back(now() - t0);
        }

        double s_vis = 0.0, s_uvw = 0.0;
        long n_flag = 0;
        for (size_t i = 0; i < nout; ++i) { s_vis += vo.get()[i].real() + vo.get()[i].imag(); n_flag += fo.get()[i] != 0; }
        for (long i = 0; i < nrow * 3; ++i) s_uvw += uo.get()[i];
        std::printf("{\"side\": \"cpp\", \"threads\": %d, \"nrow\": %ld, \"nchan_old\": %d, "
                    "\"nchan_new\": %d, \"ncorr\": %d, \"reps\": %d, "
                    "\"kernel\": %.9g, \"kernel_alloc\": %.9g, "
                    "\"kernel_min\": %.9g, \"kernel_alloc_min\": %.9g, "
                    "\"sum_vis\": %.10g, \"n_flag\": %ld, \"sum_uvw\": %.10g}\n",
                    threads, nrow, nchan_old, nchan_new, ncorr, reps,
                    median(t_kernel), median(t_alloc),
                    *std::min_element(t_kernel.begin(), t_kernel.end()),
                    *std::min_element(t_alloc.begin(), t_alloc.end()),
                    s_vis, n_flag, s_uvw);
        std::fflush(stdout);
    }
    return 0;
}
