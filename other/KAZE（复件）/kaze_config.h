/**
 * @file config.h
 * @brief Configuration file
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 * @update 2013-03-28 by Yuhua Zou
 */

#ifndef _CONFIG_H_
#define _CONFIG_H_

//******************************************************************************
//******************************************************************************

// OPENCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <omp.h>            // Enalbel OpenMP

//*************************************************************************************
//*************************************************************************************

// Some defines
#define NMAX_CHAR 400
#define HAVE_BOOST_THREADING 0 // 1: you have installed and included Boost library, 0: otherwise

// Options structure
struct toptions
{
    float soffset;          // Base scale offset (sigma units), Default: 1.60 
    int omax;               // Maximum octave evolution of the image, Default: 4. If set to 0, omax = log(min(img.rows,img.cols)) / log(2) - 2
    int nsublevels;         // Number of sublevels per scale level, Default: 4
    int img_width;          // Image width
    int img_height;         // Image height
    int diffusivity;        // Diffusivity function type, 0 -> PM G1, 1 -> PM G2 (default), 2 -> Weickert
    float sderivatives;     // Sigma smoothing derivatives, used for Gaussian smoothing
    float dthreshold;       // Detector response threshold to accept point, Default: 0.001
    float dthreshold2;      // Minimum Detector response threshold to accept point, Default: 0.00001
    bool upright;           // Use upright descriptors, not invariant to rotation, Default: false
    bool extended;          // Use extended descriptor, dimension 128, Default: false
    int descriptor;         // Descriptor Mode, 0->SURF, 1->M-SURF (default), 2->G-SURF
    bool save_scale_space;  // Default: false
    bool save_keypoints;    // Default: false
    bool verbosity;         // Default: false
    bool show_results;      // Default: false
    int nfeatures;          // Demanded number of keypoints, Default: 0

    toptions();
    toptions(std::string name, float val);
};

typedef struct
{
    cv::Mat Lx, Ly;         // 一阶微分图像（First order spatial derivatives）
    cv::Mat Lxx, Lxy, Lyy;  // 二阶微分图像（Second order spatial derivatives）
    cv::Mat Lflow;   // 传导图像（Diffusivity image）
    cv::Mat Lt;      // 进化图像（Evolution image）
    cv::Mat Lsmooth; // 平滑图像（Smoothed image）
    cv::Mat Lstep;   // 进化步长更新矩阵（Evolution step update）（！！实际未被使用！！）
    cv::Mat Ldet;    // 检测响应矩阵（Detector response）
    float etime;     // 进化时间（Evolution time）
    float esigma;    // 进化尺度（Evolution sigma. For linear diffusion t = sigma^2 / 2）
    float octave;    // 图像组（Image octave）
    float sublevel;  // 图像层级（Image sublevel in each octave）
    int sigma_size;  // 图像尺度参数的整数值，用于计算检测响应（Integer esigma. For computing the feature detector responses）
}tevolution;

// Some default options
const float DEFAULT_SCALE_OFFSET = 1.60;              // Base scale offset (sigma units)
const float DEFAULT_OCTAVE_MAX = 4.0;                 // Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
const int DEFAULT_NSUBLEVELS = 4;                     // Default number of sublevels per scale level
const float DEFAULT_DETECTOR_THRESHOLD = 0.001;       // Detector response threshold to accept point
const float DEFAULT_MIN_DETECTOR_THRESHOLD = 0.00001; // Minimum Detector response threshold to accept point
const int DEFAULT_DESCRIPTOR_MODE = 1;                // Descriptor Mode 0->SURF, 1->M-SURF
const bool DEFAULT_UPRIGHT = false;                  // Upright descriptors, not invariant to rotation
const bool DEFAULT_EXTENDED = false;                 // Extended descriptor, dimension 128
const bool DEFAULT_SAVE_SCALE_SPACE = false;         // For saving the scale space images
const bool DEFAULT_VERBOSITY = false;                // Verbosity level (0->no verbosity)
const bool DEFAULT_SHOW_RESULTS = true;              // For showing the output image with the detected features plus some ratios
const bool DEFAULT_SAVE_KEYPOINTS = false;           // For saving the list of keypoints

// Some important configuration variables
const float DEFAULT_SIGMA_SMOOTHING_DERIVATIVES = 1.0;
const float DEFAULT_KCONTRAST = .01;
const float KCONTRAST_PERCENTILE = 0.7;
const int KCONTRAST_NBINS = 300;
const bool COMPUTE_KCONTRAST = true;
const bool SUBPIXEL_REFINEMENT = true;
const int DEFAULT_DIFFUSIVITY_TYPE = 1;               // 0 -> PM G1, 1 -> PM G2, 2 -> Weickert
const bool USE_CLIPPING_NORMALIZATION = false;
const float CLIPPING_NORMALIZATION_RATIO = 1.6;
const int CLIPPING_NORMALIZATION_NITER = 5;
const float PI = 3.14159;
const float M2_PI = 6.2832;

//*************************************************************************************
//*************************************************************************************

#endif




