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
#include <fstream>
#include "utils.hpp"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

int main(int agrc, char* argv[]) {
    int height  = 4064;
    int width   = 3048;
    int type    = CV_16UC1;
    int pattern = 1; //BGGR
    int black_arr[4]    = {1024, 1024, 1024, 1024}; //BlackLevel
    int white_arr[4]    = {16512, 16512, 16512, 16512};
    float isp_gain  = 2.35;
    float WbGain[4] = {1.35, 1.0, 1.0, 1.85};
    float ccm[3][3] = {{1.0, 0.0, 0.0},
                       {0.0, 1.0, 0.0},
                       {0.0, 0.0, 1.0}};
    float gamma = 2.2;

    RawUtils *my_raw_utils_test = new RawUtils();
    Mat raw_src = my_raw_utils_test->RawRead(argv[1], height, width, type);

    Mat rggb_src = my_raw_utils_test->Bayer2Rggb(raw_src);

    Mat rggb_black_sub = my_raw_utils_test->BlackSub(rggb_src, black_arr, white_arr);

    Mat gain_rggb = my_raw_utils_test->GainAdjust(rggb_black_sub, isp_gain);

    Mat rggb_wb = my_raw_utils_test->AddWBgain(gain_rggb, WbGain);

    Mat bayer = my_raw_utils_test->Rggb2Bayer(rggb_wb);

    Mat rgb;
    cvtColor(bayer, rgb, COLOR_BayerRG2BGR_EA);

    Mat ccm_rgb = my_raw_utils_test->CcmAdjust(rgb, ccm);

    Mat gamma_rgb = my_raw_utils_test->GammaAdjust(ccm_rgb, gamma);

    imwrite("image.jpg", gamma_rgb/256);

    return 0;
}
