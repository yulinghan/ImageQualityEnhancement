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
#include "avg_blur.hpp"
#include "gaussian_blur.hpp"
#include "median_blur.hpp"
#include "bilateral_blur.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    int r = 2;
	MyAvgBlurTest *my_avg_blur_test = new MyAvgBlurTest();
    Mat avg_blur = my_avg_blur_test->Run(src, r);
    imshow("avg_blur", avg_blur);

    r = 2;
    float gauss_sigma = 3.8;
	MyGaussianBlurTest *my_gaussian_blur_test = new MyGaussianBlurTest();
    Mat gaussian_blur = my_gaussian_blur_test->Run(src, r, gauss_sigma);
    imshow("gaussian_blur", gaussian_blur);

    r = 2;
	MyMedianBlurTest *my_median_blur_test = new MyMedianBlurTest();
    Mat median_blur = my_median_blur_test->Run(src, r);
    imshow("median_blur", median_blur);

    r = 3;
    gauss_sigma = 3.8;
    float value_sigma = 13.3;
	MyBilateralBlurTest *my_bilateral_blur_test = new MyBilateralBlurTest();
    Mat bilateral_blur = my_bilateral_blur_test->Run(src, r, gauss_sigma, value_sigma);
    imshow("bilateral_blur", bilateral_blur);

    waitKey(0);

    return 0;
}
