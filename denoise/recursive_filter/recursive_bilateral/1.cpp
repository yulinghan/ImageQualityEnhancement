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
#include "recursive_bilateral.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    float sigma_spatial = 0.05;
    float sigma_range  = 0.125;

    Mat gauss_blur;
    GaussianBlur(src, gauss_blur, Size(25, 25), 0, 0);
    imshow("gauss_blur", gauss_blur);

	MyRecursiveBilateral *my_bilateral_blur_test = new MyRecursiveBilateral();
    Mat bilateral_blur = my_bilateral_blur_test->Run(src, sigma_spatial, sigma_range);
    imshow("bilateral_blur", bilateral_blur);
    waitKey(0);

    return 0;
}
