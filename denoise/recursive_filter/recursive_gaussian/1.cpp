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
#include "recursive_gaussian.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    float sigma = 3.0;
    int r = 11 * sigma;

    Mat gaussian_blur1, gaussian_blur2;
    GaussianBlur(src, gaussian_blur1, Size(r, r), 0, 0);
    imshow("GaussianBlur_blur1", gaussian_blur1);

	MyRecursiveGaussian *my_avg_blur_test = new MyRecursiveGaussian();
    gaussian_blur2 = my_avg_blur_test->Run(src, sigma*2.2);
    imshow("GaussianBlur_blur2", gaussian_blur2);
    waitKey(0);

    return 0;
}
