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
#include "sep_nlm_blur.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    float h = 8.1;
    int halfKernelSize = 3;
    int halfSearchSize = 9;
	MySepNlmBlurTest *my_sep_nlm_blur_test = new MySepNlmBlurTest();
    Mat sep_nlm_blur = my_sep_nlm_blur_test->Run(src, h, halfKernelSize, halfSearchSize);
    imshow("sep_nlm_blur", sep_nlm_blur);

    waitKey(0);

    return 0;
}
