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
#include "convpyr_poisson_test.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src1 = imread(argv[1]);
    Mat src2 = imread(argv[2]);
    Mat mask = imread(argv[3], 0);

	MyConvPyrPoissonTest *my_conv_pyr_poisson_test = new MyConvPyrPoissonTest();
    Mat out = my_conv_pyr_poisson_test->Run(src1, src2, mask);	

	out = out * 255;
	out.convertTo(out, CV_8UC1);
	imwrite(argv[4], out);

    return 0;
}

