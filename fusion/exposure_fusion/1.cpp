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
#include "exposure_fusion.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src1 = imread(argv[1]);
    Mat src2 = imread(argv[2]);
    Mat src3 = imread(argv[3]);

    vector<Mat> src_arr;
    src_arr.push_back(src1);
    src_arr.push_back(src2);
    src_arr.push_back(src3);

    MyExposureFusionTest *my_exposure_fusion_test = new MyExposureFusionTest();
    Mat out = my_exposure_fusion_test->Run(src_arr);

	imwrite(argv[4], out);

    return 0;
}
