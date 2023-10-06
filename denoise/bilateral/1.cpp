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
#include "bilateral_blur.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    int r = 15;
    float gauss_sigma = 5.8;
    float value_sigma = 10.3;
	MyBilateralBlurTest *my_bilateral_blur_test = new MyBilateralBlurTest();
    Mat bilateral_blur = my_bilateral_blur_test->Run(src, r, gauss_sigma, value_sigma);
    imshow("bilateral_blur", bilateral_blur);

    waitKey(0);

    return 0;
}
