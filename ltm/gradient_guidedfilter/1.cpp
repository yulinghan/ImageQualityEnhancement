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
#include "gradient_guidedfilter.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);
    src.convertTo(src, CV_32FC1, 1/255.0);

    int r = 5;
    float eps = 0.8;

    Mat src_gf = src.clone();
    ximgproc::guidedFilter(src, src_gf, src_gf, r, eps, -1);
    Mat p = src - src_gf;
    imshow("p", abs(p));

	MyGradientGuidedfilterTest *my_gradient_guidedfilter_test = new MyGradientGuidedfilterTest();
    Mat edge = my_gradient_guidedfilter_test->Run(src, p, r, eps);
    imshow("edge", abs(edge));

    Mat dst = src_gf + edge;
    imshow("dst1", dst);

    dst = src - edge;
    imshow("dst2", dst);
    waitKey(0);

    imwrite(argv[2], dst);

    return 0;
}
