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
#include "bilateral_ori.hpp"
#include "bilateral_lut.hpp"
#include "bilateral_pyr_blur.hpp"
#include "bilateral_separable_conv.hpp"
#include "bilateral_trigonometric.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    int r = 11;
    float gauss_sigma = 3;
    float value_sigma = 3;
/*
	MyBilateralOriTest *my_bilateral_ori_test = new MyBilateralOriTest();
    Mat bilateral_ori = my_bilateral_ori_test->Run(src, r, gauss_sigma, value_sigma);
    imshow("bilateral_ori", bilateral_ori);

    r = 15;
    gauss_sigma = 5.8;
    value_sigma = 10.3;
	MyBilateralLutTest *my_bilateral_lut_test = new MyBilateralLutTest();
    Mat bilateral_lut = my_bilateral_lut_test->Run(src, r, gauss_sigma, value_sigma);
    imshow("bilateral_lut", bilateral_lut);

    r = 9;
    gauss_sigma = 2.8;
    value_sigma = 6.3;
	MyBilateralPyrBlurTest *my_bilateral_pyr_blur_test = new MyBilateralPyrBlurTest();
    Mat bilateral_pyr_blur = my_bilateral_pyr_blur_test->Run(src, r, gauss_sigma, value_sigma);
    imshow("bilateral_pyr_blur", bilateral_pyr_blur);

    r = 15;
    gauss_sigma = 5.8;
    value_sigma = 10.3;
	MyBilateralSepConvTest *my_bilateral_sep_conv_test = new MyBilateralSepConvTest();
    Mat sep_conv = my_bilateral_sep_conv_test->Run(src, r, gauss_sigma, value_sigma);
    imshow("sep_conv", sep_conv);
*/
 
	MyBilateralTriTest *my_bilateral_tri_test = new MyBilateralTriTest();
    Mat bilateral_tri = my_bilateral_tri_test->Run(src, r, gauss_sigma, value_sigma);
    imshow("bilateral_tri", bilateral_tri);
    waitKey(0);

    return 0;
}
