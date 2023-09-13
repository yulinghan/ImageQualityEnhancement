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
#include <gsl/gsl_multimin.h>
#include "convpyr_poisson_train.hpp"

using namespace cv;
using namespace std;


int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    src.convertTo(src, CV_32FC1);
    src = src / 255.0;

	MyConvPyrPoissonTrain *my_conv_pyr_poisson_train = new MyConvPyrPoissonTrain();
    my_conv_pyr_poisson_train->Run(src);

    return 0;
}

