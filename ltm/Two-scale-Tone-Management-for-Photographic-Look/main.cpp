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
#include "two_scale_photographic.hpp"


int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    Mat ref = imread(argv[2], 0);
    imshow("src", src);
    imshow("ref", ref);

    TwoScalePhotoGraphic *my_photoGraphic_test = new TwoScalePhotoGraphic();
    Mat out = my_photoGraphic_test->Run(src, ref);
//    imshow("out", out);
    waitKey(0);

    return 0;
}
