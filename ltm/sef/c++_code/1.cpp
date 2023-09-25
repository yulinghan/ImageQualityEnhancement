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
#include "sef.hpp"
#include "exposure_fusion.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1]);

    cvtColor(src, src, COLOR_BGR2HSV);
    vector<Mat> channels;
    split(src, channels);

    MySefTest *my_sef_test = new MySefTest();
    vector<Mat> sef_arr = my_sef_test->Run(channels[2], 6.0, 0.1);

    MyExposureFusionTest *my_exposure_fusion_test = new MyExposureFusionTest();
    channels[2] = my_exposure_fusion_test->Run(sef_arr);

    Mat out;
    merge(channels, out);
    cvtColor(out, out, COLOR_HSV2BGR);

    imwrite(argv[2], out);

    return 0;
}
