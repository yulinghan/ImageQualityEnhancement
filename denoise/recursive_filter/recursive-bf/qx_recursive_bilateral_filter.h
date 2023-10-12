#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

#define QX_DEF_CHAR_MAX					255

/*Gradient domain bilateral filter*/
void qx_gradient_domain_recursive_bilateral_filter(double***out,double***in,unsigned char***texture,double sigma_spatial,double sigma_range,int h,int w,double***temp,double***temp_2w);

/*1st-order recursive bilateral filter*/
void qx_recursive_bilateral_filter(double***out,double***in,unsigned char***texture,double sigma_spatial,double sigma_range,int h,int w,double***temp,double***temp_2w,double**factor,double**temp_factor,double**temp_factor_2w);
