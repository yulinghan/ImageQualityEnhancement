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

/*
NumOfFrames:6
EVType:0
Gamma:2.2
LuxIndex:285.634
ISO:1208
Shutter:40
SensorTotalGain:12.0826
IspDigGain:1.00035
DRCGain:1.19405
Pre_ISO:1208
Pre_Shutter:40
ColorTemperature:3585
WbGain:1.32655,1,1.86722
CCM:1,0,0,0,1,0,0,0,1
BlackLevel:1023,1024,1024,1024
Orientation:90
BayerInfo(pattern:1, bitwidth:14)
ImageSize(width:4064, height:3048, stride:4064, scanline:3048)
ImageData(size:24774144, buf:0x743ec00000)
*/

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
