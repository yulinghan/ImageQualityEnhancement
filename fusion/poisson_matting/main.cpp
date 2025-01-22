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
#include "PoissonMatting.hpp"

int main(int argc, char* argv[]) {
    Mat src       = imread(argv[1], 0);
    Mat alpha_mat = imread(argv[2], 0);

    PoissonMatting *my_poissonmatting_test = new PoissonMatting();
    Mat out = my_poissonmatting_test->Run(src, alpha_mat);
    imshow("out", out);
    waitKey(0);

    return 0;
}
