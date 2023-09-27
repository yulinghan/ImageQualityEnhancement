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
#include "guidedfilter.hpp"
#include "fast_guidedfilter.hpp"
#include "weight_guidedfilter.hpp"
#include "gradient_guidedfilter.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);
    src.convertTo(src, CV_32FC1, 1/255.0);

    int r = 9;
    float eps = 0.9;

	MyGuidedfilterTest *my_guidedfilter_test = new MyGuidedfilterTest();
    Mat gf_out = my_guidedfilter_test->Run(src, src, r, eps);
    imshow("gf_out", gf_out);

    int size = 3;
	MyFastGuidedfilterTest *my_fast_guidedfilter_test = new MyFastGuidedfilterTest();
    Mat fast_gf_out = my_fast_guidedfilter_test->Run(src, src, r, eps, size);
    imshow("fast_gf_out", fast_gf_out);

	MyWeightGuidedfilterTest *my_weight_guidedfilter_test = new MyWeightGuidedfilterTest();
    Mat weight_gf_out = my_weight_guidedfilter_test->Run(src, src, r, eps);
    imshow("weight_gf_out", weight_gf_out);

	MyGradientGuidedfilterTest *my_gradient_guidedfilter_test = new MyGradientGuidedfilterTest();
    Mat gradient_gf_out = my_gradient_guidedfilter_test->Run(src, src, r, eps);
    imshow("gradient_gf_out", gradient_gf_out);

    waitKey(0);

    return 0;
}
