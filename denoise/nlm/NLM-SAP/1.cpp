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
#include "nlm_blur.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imwrite("src.png", src);
    imshow("src", src);

    float h = 5.5;
    int halfKernelSize = 7;
    int halfSearchSize = 13;
	MyNlmBlurTest *my_nlm_blur_test = new MyNlmBlurTest();
    Mat nlm_blur = my_nlm_blur_test->Run(src, h, halfKernelSize, halfSearchSize);
    imshow("nlm_blur", nlm_blur);
    imwrite(argv[2], nlm_blur);

    waitKey(0);

    return 0;
}
