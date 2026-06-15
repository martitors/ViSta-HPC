// ms_ops.cpp — versione corretta
//
// Pipeline logica in full_pipeline:
//   1. Cast vis_in complex64 -> complex128 (precisione piena)           [FIX A]
//   2. Scala UVW * 1/(1+z)  (restframe)
//   3. Ruota UVW verso nuovo phase center
//   4. Phase-shift visibilita' usando freq_old_rf e UVW rest-frame
//   5. Rebinning overlap-weighted su griglia freq_new,
//      larghezza canale per-canale (non mediana globale)                [FIX B]
//
// weight_scale NON e' calcolato qui.
// Calcolato in Python come R = df_new / (df_old_obs * (1+z)),
// identico a vista.py / CASA mstransform.                               [FIX C]
//
// full_pipeline restituisce 3 valori: (vis_out, flag_out, uvw_out).
//
// GPU DISPATCH: se compilato con -DWITH_CUDA e ms_ops_cuda.cu linkato,
// full_pipeline usa automaticamente la GPU se disponibile.
// Fallback trasparente su CPU/OpenMP se la GPU non c'è.


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <tuple>
#include <omp.h>

// ---------------------------------------------------------------------------
// GPU dispatch: dichiarazioni esterne da ms_ops_cuda.cu
// Compilate solo se -DWITH_CUDA è passato a nvcc/g++
// ---------------------------------------------------------------------------
#ifdef WITH_CUDA
extern "C" {
    int  cuda_available();
    void full_pipeline_cuda(
        const float*   vis_in_h,
        const uint8_t* flag_in_h,
        const double*  uvw_in_h,
        const double*  freq_old_h,
        const double*  freq_new_h,
        int nrow, int nchan_old, int nchan_new, int ncorr,
        double z,
        double ra_old, double dec_old,
        double ra_new, double dec_new,
        float*   vis_out_h,
        uint8_t* flag_out_h,
        double*  uvw_out_h
    );
}
static bool _gpu_available = (cuda_available() > 0);
#else
static bool _gpu_available = false;
#endif

namespace py = pybind11;

static constexpr double C_LIGHT = 299792458.0;
static constexpr double TWO_PI  = 2.0 * M_PI;

// ===========================================================================
// Helper: df mediano
// ===========================================================================
static double median_df(const double* freq, ssize_t n)
{
    if (n < 2) return 1.0;
    std::vector<double> diffs(n - 1);
    for (ssize_t i = 0; i < n - 1; ++i)
        diffs[i] = std::abs(freq[i+1] - freq[i]);
    std::nth_element(diffs.begin(), diffs.begin() + diffs.size()/2, diffs.end());
    return diffs[diffs.size()/2];
}


// ===========================================================================
// Helper: matrici di rotazione per phase shift
// ===========================================================================
static void build_basis(double ra, double dec, double M[9])
{
    double sr = std::sin(ra),  cr = std::cos(ra);
    double sd = std::sin(dec), cd = std::cos(dec);
    M[0] = -sr;    M[1] =  cr;    M[2] = 0.0;
    M[3] = -cr*sd; M[4] = -sr*sd; M[5] = cd;
    M[6] =  cr*cd; M[7] =  sr*cd; M[8] = sd;
}

static void mat_mul_BT(const double A[9], const double B[9], double R[9])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R[i*3+j] = 0.0;
            for (int k = 0; k < 3; ++k)
                R[i*3+j] += A[i*3+k] * B[j*3+k];
        }
}


// ===========================================================================
// 1. BUILD REGRID PLAN
// ===========================================================================
std::tuple<
    py::array_t<double>,
    py::array_t<int32_t>,
    py::array_t<double>,
    py::array_t<int32_t>,
    double
>
build_regrid_plan(
    py::array_t<double, py::array::c_style | py::array::forcecast> freq_old_arr,
    double factor,
    double start_hz
)
{
    auto buf       = freq_old_arr.request();
    const ssize_t nchan_old = buf.shape[0];
    const double* fo = static_cast<const double*>(buf.ptr);

    std::vector<double> diffs(nchan_old - 1);
    for (ssize_t i = 0; i < nchan_old - 1; ++i)
        diffs[i] = fo[i+1] - fo[i];
    std::nth_element(diffs.begin(), diffs.begin() + diffs.size()/2, diffs.end());
    double df_old = diffs[diffs.size()/2];
    double df_new = df_old * factor;

    double first_center = (start_hz > 0.0)
        ? start_hz
        : fo[0] - df_old / 2.0 + df_new / 2.0;

    double upper_edge   = fo[nchan_old - 1] + df_old / 2.0;
    double bw_available = upper_edge - (first_center - df_new / 2.0);
    ssize_t nchan_new   = std::max((ssize_t)std::floor(bw_available / df_new), (ssize_t)1);

    py::array_t<double>  freq_new_arr(nchan_new);
    double* fn = freq_new_arr.mutable_data();
    for (ssize_t j = 0; j < nchan_new; ++j)
        fn[j] = first_center + j * df_new;

    ssize_t nchan_int      = nchan_old;
    double  widthFactorIdx = (double)nchan_int / (double)nchan_new;

    std::vector<double> freq_int(nchan_int);
    freq_int[0] = fn[0] - df_new / 2.0;
    for (ssize_t i = 1; i < nchan_int; ++i) {
        double w_i = df_new / widthFactorIdx;
        freq_int[i] = freq_int[i-1] + w_i;
    }

    py::array_t<int32_t> chan_map_arr(nchan_int);
    int32_t* cm = chan_map_arr.mutable_data();
    for (ssize_t k = 0; k < nchan_int; ++k) {
        double f = freq_int[k];
        ssize_t j = (ssize_t)std::floor((f - (fn[0] - df_new / 2.0)) / df_new);
        cm[k] = (j >= 0 && j < nchan_new) ? (int32_t)j : -1;
    }

    py::array_t<int32_t> indices_arr({nchan_int, (ssize_t)2});
    py::array_t<double>  weights_arr(nchan_int);
    int32_t* idx = indices_arr.mutable_data();
    double*  wgt = weights_arr.mutable_data();

    for (ssize_t k = 0; k < nchan_int; ++k) {
        double f = freq_int[k];
        ssize_t pos = (ssize_t)(std::lower_bound(fo, fo + nchan_old, f) - fo) - 1;
        pos = std::max((ssize_t)0, std::min(pos, nchan_old - 2));

        bool oob = (f < fo[0]) || (f > fo[nchan_old - 1]);
        if (oob) {
            idx[k*2+0] = -1; idx[k*2+1] = -1; wgt[k] = 0.0;
        } else {
            idx[k*2+0] = (int32_t)pos;
            idx[k*2+1] = (int32_t)(pos + 1);
            double span = fo[pos+1] - fo[pos];
            wgt[k] = (span > 0.0) ? std::min(std::max((f - fo[pos]) / span, 0.0), 1.0) : 0.0;
        }
    }

    return {freq_new_arr, indices_arr, weights_arr, chan_map_arr, df_new};
}


// ===========================================================================
// 2. REGRID KERNEL (standalone, usato da regrid_kernel e regrid_and_shift)
// ===========================================================================
std::pair<py::array_t<std::complex<double>>, py::array_t<bool>>
regrid_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> vis_in,
    py::array_t<bool,                  py::array::c_style | py::array::forcecast> flag_in,
    py::array_t<double,                py::array::c_style | py::array::forcecast> freq_old_arr,
    py::array_t<double,                py::array::c_style | py::array::forcecast> freq_new_arr
)
{
    auto vbuf   = vis_in.request();
    auto fbuf_o = freq_old_arr.request();
    auto fbuf_n = freq_new_arr.request();

    const ssize_t nrow      = vbuf.shape[0];
    const ssize_t nchan_old = vbuf.shape[1];
    const ssize_t ncorr     = vbuf.shape[2];
    const ssize_t nchan_new = fbuf_n.shape[0];

    const double* fo = static_cast<const double*>(fbuf_o.ptr);
    const double* fn = static_cast<const double*>(fbuf_n.ptr);

    double df_old = median_df(fo, nchan_old);
    double df_new = median_df(fn, nchan_new);

    py::array_t<std::complex<double>> vis_out({nrow, nchan_new, ncorr});
    py::array_t<bool>                  flag_out({nrow, nchan_new, ncorr});

    auto vout = vis_out.mutable_unchecked<3>();
    auto fout = flag_out.mutable_unchecked<3>();
    auto vin  = vis_in.unchecked<3>();
    auto fin  = flag_in.unchecked<3>();

    std::vector<std::complex<double>> accum(nchan_new);
    std::vector<double>               out_wgt(nchan_new);

    for (ssize_t i = 0; i < nrow; ++i) {
        for (ssize_t c = 0; c < ncorr; ++c) {
            std::fill(accum.begin(),   accum.end(),   std::complex<double>(0.0, 0.0));
            std::fill(out_wgt.begin(), out_wgt.end(), 0.0);

            for (ssize_t k = 0; k < nchan_old; ++k) {
                if ((bool)fin(i, k, c)) continue;
                double lo_in = fo[k] - df_old * 0.5;
                double hi_in = fo[k] + df_old * 0.5;
                ssize_t j0 = std::max((ssize_t)0,
                    (ssize_t)std::floor((lo_in - (fn[0] + df_new * 0.5)) / df_new));
                ssize_t j1 = std::min(nchan_new - 1,
                    (ssize_t)std::floor((hi_in - (fn[0] - df_new * 0.5)) / df_new));
                for (ssize_t j = j0; j <= j1; ++j) {
                    double lo_out = fn[j] - df_new * 0.5;
                    double hi_out = fn[j] + df_new * 0.5;
                    double overlap = std::max(0.0,
                        std::min(hi_in, hi_out) - std::max(lo_in, lo_out));
                    double w = overlap / df_old;
                    if (w <= 0.0) continue;
                    accum[j]   += vin(i, k, c) * w;
                    out_wgt[j] += w;
                }
            }
            for (ssize_t j = 0; j < nchan_new; ++j) {
                if (out_wgt[j] > 0.0) {
                    vout(i, j, c) = accum[j] / out_wgt[j];
                    fout(i, j, c) = false;
                } else {
                    vout(i, j, c) = {0.0, 0.0};
                    fout(i, j, c) = true;
                }
            }
        }
    }
    return {vis_out, flag_out};
}


// ===========================================================================
// 3. PHASE SHIFT
// ===========================================================================
std::pair<py::array_t<std::complex<double>>, py::array_t<double>>
phase_shift(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> vis_in,
    py::array_t<double,                py::array::c_style | py::array::forcecast> uvw_in,
    py::array_t<double,                py::array::c_style | py::array::forcecast> freq_arr,
    double ra_old, double dec_old, double ra_new, double dec_new
)
{
    auto vbuf = vis_in.request();
    auto fbuf = freq_arr.request();

    const ssize_t nrow  = vbuf.shape[0];
    const ssize_t nchan = vbuf.shape[1];
    const ssize_t ncorr = vbuf.shape[2];
    const double* freq  = static_cast<const double*>(fbuf.ptr);

    bool equispaced = true;
    double df = 0.0;
    if (nchan > 1) {
        df = freq[1] - freq[0];
        for (ssize_t j = 2; j < nchan; ++j)
            if (std::abs((freq[j] - freq[j-1]) - df) > df * 1e-6) { equispaced = false; break; }
    }

    double basis_old[9], basis_new[9], Rot[9];
    build_basis(ra_old, dec_old, basis_old);
    build_basis(ra_new, dec_new, basis_new);
    mat_mul_BT(basis_new, basis_old, Rot);

    double dl  = (ra_new - ra_old) * std::cos(dec_old);
    double dm  = (dec_new - dec_old);
    double arg = 1.0 - dl*dl - dm*dm;
    double dn  = (arg > 0.0 ? std::sqrt(arg) : 0.0) - 1.0;

    py::array_t<std::complex<double>> vis_out({nrow, nchan, ncorr});
    py::array_t<double>                uvw_out({nrow, (ssize_t)3});

    auto vout = vis_out.mutable_unchecked<3>();
    auto uout = uvw_out.mutable_unchecked<2>();
    auto vin  = vis_in.unchecked<3>();
    auto uin  = uvw_in.unchecked<2>();

    for (ssize_t i = 0; i < nrow; ++i) {
        double u = uin(i,0), v = uin(i,1), w = uin(i,2);
        uout(i,0) = Rot[0]*u + Rot[1]*v + Rot[2]*w;
        uout(i,1) = Rot[3]*u + Rot[4]*v + Rot[5]*w;
        uout(i,2) = Rot[6]*u + Rot[7]*v + Rot[8]*w;
        double proj = u*dl + v*dm + w*dn;
        if (equispaced && nchan > 1) {
            double phi0 = -TWO_PI * proj * freq[0] / C_LIGHT;
            double dphi = -TWO_PI * proj * df      / C_LIGHT;
            std::complex<double> ph   = std::polar(1.0, phi0);
            std::complex<double> step = std::polar(1.0, dphi);
            for (ssize_t j = 0; j < nchan; ++j) {
                if (j % 64 == 0) ph = std::polar(1.0, phi0 + j * dphi);
                for (ssize_t c = 0; c < ncorr; ++c) vout(i, j, c) = vin(i, j, c) * ph;
                ph *= step;
            }
        } else {
            for (ssize_t j = 0; j < nchan; ++j) {
                double phi = -TWO_PI * proj * freq[j] / C_LIGHT;
                std::complex<double> ph = std::polar(1.0, phi);
                for (ssize_t c = 0; c < ncorr; ++c) vout(i, j, c) = vin(i, j, c) * ph;
            }
        }
    }
    return {vis_out, uvw_out};
}


// ===========================================================================
// 4. REGRID_AND_SHIFT
// ===========================================================================
std::tuple<
    py::array_t<std::complex<float>>,
    py::array_t<bool>,
    py::array_t<double>
>
regrid_and_shift(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> vis_in,
    py::array_t<bool,                  py::array::c_style | py::array::forcecast> flag_in,
    py::array_t<double,                py::array::c_style | py::array::forcecast> uvw_in,
    py::array_t<double,                py::array::c_style | py::array::forcecast> freq_old_arr,
    py::array_t<double,                py::array::c_style | py::array::forcecast> freq_new_arr,
    double ra_old, double dec_old, double ra_new, double dec_new
)
{
    auto vbuf   = vis_in.request();
    auto fbuf_o = freq_old_arr.request();
    auto fbuf_n = freq_new_arr.request();

    const ssize_t nrow      = vbuf.shape[0];
    const ssize_t nchan_old = vbuf.shape[1];
    const ssize_t ncorr     = vbuf.shape[2];
    const ssize_t nchan_new = fbuf_n.shape[0];

    const double* fo = static_cast<const double*>(fbuf_o.ptr);
    const double* fn = static_cast<const double*>(fbuf_n.ptr);

    double df_old = median_df(fo, nchan_old);
    double df_new = median_df(fn, nchan_new);

    bool equispaced = (nchan_new > 1);
    for (ssize_t j = 2; j < nchan_new && equispaced; ++j)
        if (std::abs((fn[j]-fn[j-1])-(fn[1]-fn[0])) > (fn[1]-fn[0])*1e-6) equispaced = false;

    double basis_old[9], basis_new[9], Rot[9];
    build_basis(ra_old, dec_old, basis_old);
    build_basis(ra_new, dec_new, basis_new);
    mat_mul_BT(basis_new, basis_old, Rot);

    double dl = (ra_new-ra_old)*std::cos(dec_old), dm = dec_new-dec_old;
    double arg = 1.0-dl*dl-dm*dm;
    double dn = (arg>0.0?std::sqrt(arg):0.0)-1.0;

    py::array_t<std::complex<float>> vis_out({nrow,nchan_new,ncorr});
    py::array_t<bool>                  flag_out({nrow,nchan_new,ncorr});
    py::array_t<double>                uvw_out({nrow,(ssize_t)3});

    auto vout=vis_out.mutable_unchecked<3>(); auto fout=flag_out.mutable_unchecked<3>();
    auto uout=uvw_out.mutable_unchecked<2>();
    auto vin=vis_in.unchecked<3>(); auto fin=flag_in.unchecked<3>(); auto uin=uvw_in.unchecked<2>();

    std::vector<std::complex<double>> accum(nchan_new), vis_rg(nchan_new*ncorr);
    std::vector<double> out_wgt(nchan_new);
    std::vector<uint8_t> flg_rg(nchan_new*ncorr);

    for (ssize_t i = 0; i < nrow; ++i) {
        for (ssize_t c = 0; c < ncorr; ++c) {
            std::fill(accum.begin(),accum.end(),std::complex<double>(0.0,0.0));
            std::fill(out_wgt.begin(),out_wgt.end(),0.0);
            for (ssize_t k = 0; k < nchan_old; ++k) {
                if ((bool)fin(i,k,c)) continue;
                double lo_in=fo[k]-df_old*0.5, hi_in=fo[k]+df_old*0.5;
                ssize_t j0=std::max((ssize_t)0,(ssize_t)std::floor((lo_in-(fn[0]+df_new*0.5))/df_new));
                ssize_t j1=std::min(nchan_new-1,(ssize_t)std::floor((hi_in-(fn[0]-df_new*0.5))/df_new));
                for (ssize_t j=j0;j<=j1;++j) {
                    double ov=std::max(0.0,std::min(fo[k]+df_old*0.5,fn[j]+df_new*0.5)
                                         -std::max(fo[k]-df_old*0.5,fn[j]-df_new*0.5));
                    double w=ov/df_old; if(w<=0.0) continue;
                    accum[j]+=vin(i,k,c)*w; out_wgt[j]+=w;
                }
            }
            for (ssize_t j=0;j<nchan_new;++j) {
                if(out_wgt[j]>0.0){vis_rg[j*ncorr+c]=accum[j]/out_wgt[j];flg_rg[j*ncorr+c]=0;}
                else{vis_rg[j*ncorr+c]={0.0,0.0};flg_rg[j*ncorr+c]=1;}
            }
        }
        double u=uin(i,0),v=uin(i,1),w=uin(i,2);
        uout(i,0)=Rot[0]*u+Rot[1]*v+Rot[2]*w;
        uout(i,1)=Rot[3]*u+Rot[4]*v+Rot[5]*w;
        uout(i,2)=Rot[6]*u+Rot[7]*v+Rot[8]*w;
        double proj=u*dl+v*dm+w*dn;
        double df_n=(nchan_new>1)?(fn[1]-fn[0]):0.0;
        if (equispaced && nchan_new>1) {
            double phi0=-TWO_PI*proj*fn[0]/C_LIGHT, dphi=-TWO_PI*proj*df_n/C_LIGHT;
            std::complex<double> ph=std::polar(1.0,phi0), step=std::polar(1.0,dphi);
            for (ssize_t j=0;j<nchan_new;++j) {
                if(j%64==0) ph=std::polar(1.0,phi0+j*dphi);
                for(ssize_t c=0;c<ncorr;++c){vout(i,j,c)=vis_rg[j*ncorr+c]*ph;fout(i,j,c)=(bool)flg_rg[j*ncorr+c];}
                ph*=step;
            }
        } else {
            for(ssize_t j=0;j<nchan_new;++j){
                double phi=-TWO_PI*proj*fn[j]/C_LIGHT;
                std::complex<double> ph=std::polar(1.0,phi);
                for(ssize_t c=0;c<ncorr;++c){vout(i,j,c)=vis_rg[j*ncorr+c]*ph;fout(i,j,c)=(bool)flg_rg[j*ncorr+c];}
            }
        }
    }
    return {vis_out,flag_out,uvw_out};
}


// ===========================================================================
// 5. RESTFRAME_UVW
// ===========================================================================
py::array_t<double>
restframe_uvw(py::array_t<double,py::array::c_style|py::array::forcecast> uvw_in, double z)
{
    auto buf=uvw_in.request(); const ssize_t nrow=buf.shape[0]; const double inv=1.0/(1.0+z);
    py::array_t<double> uvw_out({nrow,(ssize_t)3});
    const double* src=static_cast<const double*>(buf.ptr); double* dst=uvw_out.mutable_data();
    for(ssize_t i=0;i<nrow*3;++i) dst[i]=src[i]*inv;
    return uvw_out;
}


// ===========================================================================
// 6. SCALE_ARRAY
// ===========================================================================
py::array_t<double>
scale_array(py::array_t<double,py::array::c_style|py::array::forcecast> arr_in, double factor)
{
    auto buf=arr_in.request(); py::array_t<double> arr_out(buf.shape);
    const double* src=static_cast<const double*>(buf.ptr); double* dst=arr_out.mutable_data();
    ssize_t N=1; for(auto s:buf.shape) N*=s;
    for(ssize_t i=0;i<N;++i) dst[i]=src[i]*factor;
    return arr_out;
}


// ===========================================================================
// 7. RESTFRAME_AND_SHIFT
// ===========================================================================
std::pair<py::array_t<std::complex<double>>, py::array_t<double>>
restframe_and_shift(
    py::array_t<std::complex<double>,py::array::c_style|py::array::forcecast> vis_in,
    py::array_t<double,py::array::c_style|py::array::forcecast> uvw_in,
    py::array_t<double,py::array::c_style|py::array::forcecast> freq_arr,
    double z, double ra_old, double dec_old, double ra_new, double dec_new
)
{
    auto vbuf=vis_in.request(); auto fbuf=freq_arr.request();
    const ssize_t nrow=vbuf.shape[0], nchan=vbuf.shape[1], ncorr=vbuf.shape[2];
    const double* freq=static_cast<const double*>(fbuf.ptr);
    const double inv_z=1.0/(1.0+z);
    bool equispaced=true; double df=0.0;
    if(nchan>1){df=freq[1]-freq[0]; for(ssize_t j=2;j<nchan;++j) if(std::abs((freq[j]-freq[j-1])-df)>df*1e-6){equispaced=false;break;}}
    double basis_old[9],basis_new[9],Rot[9];
    build_basis(ra_old,dec_old,basis_old); build_basis(ra_new,dec_new,basis_new); mat_mul_BT(basis_new,basis_old,Rot);
    double dl=(ra_new-ra_old)*std::cos(dec_old), dm=dec_new-dec_old;
    double arg=1.0-dl*dl-dm*dm; double dn=(arg>0.0?std::sqrt(arg):0.0)-1.0;
    py::array_t<std::complex<double>> vis_out({nrow,nchan,ncorr});
    py::array_t<double> uvw_out({nrow,(ssize_t)3});
    auto vout=vis_out.mutable_unchecked<3>(); auto uout=uvw_out.mutable_unchecked<2>();
    auto vin=vis_in.unchecked<3>(); auto uin=uvw_in.unchecked<2>();
    for(ssize_t i=0;i<nrow;++i){
        double u_rf=uin(i,0)*inv_z, v_rf=uin(i,1)*inv_z, w_rf=uin(i,2)*inv_z;
        uout(i,0)=Rot[0]*u_rf+Rot[1]*v_rf+Rot[2]*w_rf;
        uout(i,1)=Rot[3]*u_rf+Rot[4]*v_rf+Rot[5]*w_rf;
        uout(i,2)=Rot[6]*u_rf+Rot[7]*v_rf+Rot[8]*w_rf;
        double proj=u_rf*dl+v_rf*dm+w_rf*dn;
        if(equispaced&&nchan>1){
            double phi0=-TWO_PI*proj*freq[0]/C_LIGHT, dphi=-TWO_PI*proj*df/C_LIGHT;
            std::complex<double> ph=std::polar(1.0,phi0), step=std::polar(1.0,dphi);
            for(ssize_t j=0;j<nchan;++j){if(j%64==0)ph=std::polar(1.0,phi0+j*dphi);for(ssize_t c=0;c<ncorr;++c)vout(i,j,c)=vin(i,j,c)*ph;ph*=step;}
        } else {
            for(ssize_t j=0;j<nchan;++j){double phi=-TWO_PI*proj*freq[j]/C_LIGHT;std::complex<double> ph=std::polar(1.0,phi);for(ssize_t c=0;c<ncorr;++c)vout(i,j,c)=vin(i,j,c)*ph;}
        }
    }
    return {vis_out,uvw_out};
}


// ===========================================================================
// 8. FULL_PIPELINE — VERSIONE CORRETTA
//
// Modifiche rispetto alla versione originale:
//
//  A) vis_in e' accettato come complex64 (come arriva da casacore getcolslice)
//     ma viene convertito a complex128 subito, PRIMA del phase-shift.
//     Evita perdita di precisione nella fase.
//
//  B) Il rebinning calcola df_old_k PER CANALE (non la mediana globale):
//       df_old_k[k] = (freq_old_rf[k+1] - freq_old_rf[k-1]) / 2   (differenza centrata)
//     con casi border per k=0 e k=nchan_old-1. Identico a CASA mstransform.
//
//  C) weight_scale NON viene piu' calcolato qui. Viene calcolato in Python
//     come R = df_new / df_old_obs (larghezza canale letta dalla SPECTRAL_WINDOW
//     table), identico a vista.py / CASA mstransform. Stabile, deterministico,
//     indipendente dai flag e dai bordi della griglia.
//
//  D) interp_indices, interp_weights, chan_map sono RIMOSSI dalla firma.
//     Erano passati ma ignorati — fonte di confusione.
//
// Output:
//   vis_out   complex64   (nrow, nchan_new, ncorr)
//   flag_out  bool        (nrow, nchan_new, ncorr)
//   uvw_out   float64     (nrow, 3)
// ===========================================================================
std::tuple<
    py::array_t<std::complex<float>>,
    py::array_t<bool>,
    py::array_t<double>
>
full_pipeline(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> vis_in_f32,
    py::array_t<bool,                py::array::c_style | py::array::forcecast> flag_in,
    py::array_t<double,              py::array::c_style | py::array::forcecast> uvw_in,
    double z,
    py::array_t<double,              py::array::c_style | py::array::forcecast> freq_old_rf_arr,
    double ra_old, double dec_old,
    double ra_new, double dec_new,
    py::array_t<double,              py::array::c_style | py::array::forcecast> freq_new_arr
)
{
    // ------------------------------------------------------------------ dims
    auto vbuf   = vis_in_f32.request();
    auto fbuf_o = freq_old_rf_arr.request();
    auto fbuf_n = freq_new_arr.request();

    const ssize_t nrow      = vbuf.shape[0];
    const ssize_t nchan_old = vbuf.shape[1];
    const ssize_t ncorr     = vbuf.shape[2];
    const ssize_t nchan_new = fbuf_n.shape[0];

    const double* freq_old_rf = static_cast<const double*>(fbuf_o.ptr);
    const double* freq_new    = static_cast<const double*>(fbuf_n.ptr);

    // ----------------------------------------------------------------
    // GPU DISPATCH: se GPU disponibile, delega a full_pipeline_cuda e
    // ritorna direttamente. Fallback trasparente su CPU/OpenMP sotto.
    // ----------------------------------------------------------------
#ifdef WITH_CUDA
    if (_gpu_available) {
        // Prepara output arrays
        py::array_t<std::complex<float>> vis_out_gpu({nrow, nchan_new, ncorr});
        py::array_t<bool>                flag_out_gpu({nrow, nchan_new, ncorr});
        py::array_t<double>              uvw_out_gpu({nrow, (ssize_t)3});

        full_pipeline_cuda(
            reinterpret_cast<const float*>(vbuf.ptr),         // complex64 → float*
            static_cast<const uint8_t*>(flag_in.request().ptr),
            static_cast<const double*>(uvw_in.request().ptr),
            freq_old_rf, freq_new,
            (int)nrow, (int)nchan_old, (int)nchan_new, (int)ncorr,
            z, ra_old, dec_old, ra_new, dec_new,
            reinterpret_cast<float*>(vis_out_gpu.mutable_data()),
            reinterpret_cast<uint8_t*>(flag_out_gpu.mutable_data()),
            uvw_out_gpu.mutable_data()
        );
        return {vis_out_gpu, flag_out_gpu, uvw_out_gpu};
    }
#endif

    // ----------------------------------------------------------------
    // FIX A: converti vis_in complex64 -> complex128 PRIMA di qualsiasi
    //         operazione. Evita perdita di precisione nella fase.
    // ----------------------------------------------------------------
    const std::complex<float>* vin_f32_ptr =
        static_cast<const std::complex<float>*>(vbuf.ptr);
    // FIX A (ottimizzato): NESSUN buffer complex128 da ~1 GB e NESSUNA
    // passata seriale di conversione. Il cast float->double avviene
    // per-elemento dentro il loop OpenMP, dove il dato serve, senza
    // alcuna perdita di precisione (si casta prima della moltiplicazione
    // per la fase, identico a prima).

    // ----------------------------------------------------------------
    // FIX B: larghezza canale PER-CANALE invece della mediana globale.
    // df_old_k[k] = larghezza del k-esimo canale input (differenza centrata).
    // ----------------------------------------------------------------
    std::vector<double> df_old_k(nchan_old);
    if (nchan_old == 1) {
        df_old_k[0] = (nchan_new > 1) ? (freq_new[1] - freq_new[0]) : 1.0;
    } else {
        df_old_k[0] = freq_old_rf[1] - freq_old_rf[0];
        for (ssize_t k = 1; k < nchan_old - 1; ++k)
            df_old_k[k] = (freq_old_rf[k+1] - freq_old_rf[k-1]) * 0.5;
        df_old_k[nchan_old-1] = freq_old_rf[nchan_old-1] - freq_old_rf[nchan_old-2];
    }
    // Assicura valori positivi (gestisce spw con freq decrescenti)
    for (ssize_t k = 0; k < nchan_old; ++k)
        df_old_k[k] = std::abs(df_old_k[k]);

    double df_new = median_df(freq_new, nchan_new);

    // ----------------------------------------------------------------
    // Calcola equispacing griglia old (per ottimizzazione phase-shift)
    // ----------------------------------------------------------------
    bool eq_old = (nchan_old > 1);
    double df_old_eq = (nchan_old > 1) ? (freq_old_rf[1] - freq_old_rf[0]) : 0.0;
    for (ssize_t j = 2; j < nchan_old && eq_old; ++j)
        if (std::abs((freq_old_rf[j]-freq_old_rf[j-1])-df_old_eq) > std::abs(df_old_eq)*1e-6)
            eq_old = false;

    // ----------------------------------------------------------------
    // Matrici di rotazione e proiezione per phase-shift + UVW
    // ----------------------------------------------------------------
    double basis_old[9], basis_new_m[9], Rot[9];
    build_basis(ra_old, dec_old, basis_old);
    build_basis(ra_new, dec_new, basis_new_m);
    mat_mul_BT(basis_new_m, basis_old, Rot);
    double dl  = (ra_new - ra_old) * std::cos(dec_old);
    double dm  = (dec_new - dec_old);
    double arg = 1.0 - dl*dl - dm*dm;
    double dn  = (arg > 0.0 ? std::sqrt(arg) : 0.0) - 1.0;
    const double inv_z = 1.0 / (1.0 + z);

    // ----------------------------------------------------------------
    // Output
    // ----------------------------------------------------------------
    py::array_t<std::complex<float>> vis_out({nrow, nchan_new, ncorr});
    py::array_t<bool>                flag_out({nrow, nchan_new, ncorr});
    py::array_t<double>              uvw_out({nrow, (ssize_t)3});

    auto vout = vis_out.mutable_unchecked<3>();
    auto fout = flag_out.mutable_unchecked<3>();
    auto uout = uvw_out.mutable_unchecked<2>();
    auto fin  = flag_in.unchecked<3>();
    auto uin  = uvw_in.unchecked<2>();

    // OpenMP: ogni riga e' completamente indipendente.
#pragma omp parallel for schedule(dynamic, 64) default(shared)
    for (ssize_t i = 0; i < nrow; ++i) {
        // Buffer privati per thread (stack, non heap allocation per riga piccola)
        std::vector<std::complex<double>> vis_shifted(nchan_old * ncorr);
        std::vector<std::complex<double>> accum(nchan_new);
        std::vector<double>               out_wgt(nchan_new);

        // --- 1. Restframe UVW ---
        double u_rf = uin(i,0) * inv_z;
        double v_rf = uin(i,1) * inv_z;
        double w_rf = uin(i,2) * inv_z;

        // --- 2. Ruota UVW ---
        uout(i,0) = Rot[0]*u_rf + Rot[1]*v_rf + Rot[2]*w_rf;
        uout(i,1) = Rot[3]*u_rf + Rot[4]*v_rf + Rot[5]*w_rf;
        uout(i,2) = Rot[6]*u_rf + Rot[7]*v_rf + Rot[8]*w_rf;
        double proj = u_rf*dl + v_rf*dm + w_rf*dn;

        // --- 3. Phase shift su freq_old_rf (rest-frame) a precision float64 ---
        // FIX A: cast float32->float64 inline (vedi sopra), niente buffer globale
        if (eq_old && nchan_old > 1) {
            double phi0 = -TWO_PI * proj * freq_old_rf[0] / C_LIGHT;
            double dphi = -TWO_PI * proj * df_old_eq      / C_LIGHT;
            std::complex<double> ph   = std::polar(1.0, phi0);
            std::complex<double> step = std::polar(1.0, dphi);
            for (ssize_t j = 0; j < nchan_old; ++j) {
                if (j % 64 == 0) ph = std::polar(1.0, phi0 + j * dphi);
                for (ssize_t c = 0; c < ncorr; ++c)
                    vis_shifted[j*ncorr+c] = static_cast<std::complex<double>>(
                        vin_f32_ptr[(i*nchan_old+j)*ncorr+c]) * ph;
                ph *= step;
            }
        } else {
            for (ssize_t j = 0; j < nchan_old; ++j) {
                double phi = -TWO_PI * proj * freq_old_rf[j] / C_LIGHT;
                std::complex<double> ph = std::polar(1.0, phi);
                for (ssize_t c = 0; c < ncorr; ++c)
                    vis_shifted[j*ncorr+c] = static_cast<std::complex<double>>(
                        vin_f32_ptr[(i*nchan_old+j)*ncorr+c]) * ph;
            }
        }

        // --- 4. Rebinning overlap con larghezza canale PER-CANALE ---
        for (ssize_t c = 0; c < ncorr; ++c) {
            std::fill(accum.begin(),   accum.end(),   std::complex<double>(0.0, 0.0));
            std::fill(out_wgt.begin(), out_wgt.end(), 0.0);

            for (ssize_t k = 0; k < nchan_old; ++k) {
                if ((bool)fin(i, k, c)) continue;

                // FIX B: usa df_old_k[k] per questo canale specifico
                double dfo  = df_old_k[k];
                double lo_in = freq_old_rf[k] - dfo * 0.5;
                double hi_in = freq_old_rf[k] + dfo * 0.5;

                ssize_t j0 = std::max((ssize_t)0,
                    (ssize_t)std::floor((lo_in - (freq_new[0] + df_new * 0.5)) / df_new));
                ssize_t j1 = std::min(nchan_new - 1,
                    (ssize_t)std::floor((hi_in - (freq_new[0] - df_new * 0.5)) / df_new));

                for (ssize_t j = j0; j <= j1; ++j) {
                    double lo_out = freq_new[j] - df_new * 0.5;
                    double hi_out = freq_new[j] + df_new * 0.5;
                    double ov = std::max(0.0,
                        std::min(hi_in, hi_out) - std::max(lo_in, lo_out));
                    // FIX B: normalizza per df_old_k[k] (larghezza del canale sorgente)
                    double w = ov / dfo;
                    if (w <= 0.0) continue;
                    accum[j]   += vis_shifted[k*ncorr+c] * w;
                    out_wgt[j] += w;
                }
            }

            for (ssize_t j = 0; j < nchan_new; ++j) {
                if (out_wgt[j] > 0.0) {
                    vout(i, j, c) = static_cast<std::complex<float>>(accum[j] / out_wgt[j]);
                    fout(i, j, c) = false;
                } else {
                    vout(i, j, c) = {0.0f, 0.0f};
                    fout(i, j, c) = true;
                }
            }
        }
    }

    return {vis_out, flag_out, uvw_out};
}


// ===========================================================================
// PYBIND11
// ===========================================================================
PYBIND11_MODULE(ms_ops, m)
{
    m.def("build_regrid_plan", &build_regrid_plan,
          py::arg("freq_old"), py::arg("factor"), py::arg("start_hz") = -1.0);

    m.def("regrid_kernel", &regrid_kernel,
          py::arg("vis"), py::arg("flag"), py::arg("freq_old"), py::arg("freq_new"));

    m.def("phase_shift", &phase_shift,
          py::arg("vis"), py::arg("uvw"), py::arg("freq"),
          py::arg("ra_old"), py::arg("dec_old"), py::arg("ra_new"), py::arg("dec_new"));

    m.def("regrid_and_shift", &regrid_and_shift,
          py::arg("vis"), py::arg("flag"), py::arg("uvw"),
          py::arg("freq_old"), py::arg("freq_new"),
          py::arg("ra_old"), py::arg("dec_old"), py::arg("ra_new"), py::arg("dec_new"));

    m.def("restframe_uvw", &restframe_uvw,
          py::arg("uvw"), py::arg("z"));

    m.def("scale_array", &scale_array,
          py::arg("arr"), py::arg("factor"));

    m.def("restframe_and_shift", &restframe_and_shift,
          py::arg("vis"), py::arg("uvw"), py::arg("freq_old_rf"), py::arg("z"),
          py::arg("ra_old"), py::arg("dec_old"), py::arg("ra_new"), py::arg("dec_new"));

    m.def("full_pipeline", &full_pipeline,
          py::arg("vis"), py::arg("flag"), py::arg("uvw"), py::arg("z"),
          py::arg("freq_old_rf"),
          py::arg("ra_old"), py::arg("dec_old"), py::arg("ra_new"), py::arg("dec_new"),
          py::arg("freq_new"),
          "Pipeline completa corretta: restframe -> centering -> rebinning overlap. "
          "Restituisce (vis_out, flag_out, uvw_out). "
          "vis_in complex64 convertito a float64 prima del phase-shift. "
          "weight_scale calcolato in Python come R = df_new / df_old_obs.");
}

