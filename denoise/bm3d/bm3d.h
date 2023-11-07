#ifndef BM3D_H_INCLUDED
#define BM3D_H_INCLUDED

#include <fftw3.h>
#include <vector>

/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Main function
int run_bm3d(
    const float sigma
,   std::vector<float> &img_noisy
,   std::vector<float> &img_basic
,   std::vector<float> &img_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned patch_size = 0
);

//! 1st step of BM3D
void bm3d_1st_step(
    const float sigma
,   std::vector<float> const& img_noisy
,   std::vector<float> &img_basic
,   const unsigned width
,   const unsigned height
,   const unsigned nHard
,   const unsigned kHard
,   const unsigned NHard
,   const unsigned pHard
,   fftwf_plan *  plan_2d_for_1
,   fftwf_plan *  plan_2d_for_2
,   fftwf_plan *  plan_2d_inv
);

//! 2nd step of BM3D
void bm3d_2nd_step(
    const float sigma
,   std::vector<float> const& img_noisy
,   std::vector<float> const& img_basic
,   std::vector<float> &img_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned nWien
,   const unsigned kWien
,   const unsigned NWien
,   const unsigned pWien
,   fftwf_plan *  plan_2d_for_1
,   fftwf_plan *  plan_2d_for_2
,   fftwf_plan *  plan_2d_inv
);

//! Process 2D dct of a group of patches
void dct_2d_process(
    std::vector<float> &DCT_table_2D
,   std::vector<float> const& img
,   fftwf_plan * plan_1
,   fftwf_plan * plan_2
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   std::vector<float> const& coef_norm
,   const unsigned i_min
,   const unsigned i_max
);

void dct_2d_inverse(
    std::vector<float> &group_3D_table
,   const unsigned kHW
,   const unsigned N
,   std::vector<float> const& coef_norm_inv
,   fftwf_plan * plan
);

//! HT filtering using Welsh-Hadamard transform (do only
//! third dimension transform, Hard Thresholding
//! and inverse Hadamard transform)
void ht_filtering_hadamard(
    std::vector<float> &group_3D
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kHard
,   float const& sigma
,   const float lambdaThr3D
,   float &weight
);

//! Wiener filtering using Welsh-Hadamard transform
void wiener_filtering_hadamard(
    std::vector<float> &group_3D_img
,   std::vector<float> &group_3D_est
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kWien
,   float const& sigma
,   float &weight
);

/** ---------------------------------- **/
/** - Preprocessing / Postprocessing - **/
/** ---------------------------------- **/
//! Preprocess coefficients of the Kaiser window and normalization coef for the DCT
void preProcess(
    std::vector<float> &kaiserWindow
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned kHW
);

void precompute_BM(
    std::vector<std::vector<unsigned> > &patch_table
,   const std::vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned n
,   const unsigned pHW
,   const float    tauMatch
);

void hadamard_transform(
    std::vector<float> &vec
,   std::vector<float> &tmp
,   const unsigned N
,   const unsigned D
);

#endif // BM3D_H_INCLUDED
