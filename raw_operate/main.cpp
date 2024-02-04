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

Mat Rgb2Raw(Mat src) {
    int height  = 640;
    int width   = 437;
    int type    = CV_16UC1;
    int pattern = 1; //BGGR
    int black_arr[4]    = {1024, 1024, 1024, 1024}; //BlackLevel
    int white_arr[4]    = {16512, 16512, 16512, 16512};
    float isp_gain  = 1.0/1.37;
    float WbGain[4] = {1/1.55, 1.0, 1.0, 1/1.36};
    float ccm[3][3] = {{1.0, 0.0, 0.0},
                       {0.0, 1.0, 0.0},
                       {0.0, 0.0, 1.0}};
    float gamma = 1;

    RawUtils *my_raw_utils_test = new RawUtils();

    Mat gamma_rgb = my_raw_utils_test->GammaAdjust(src, gamma);

    Mat ccm_rgb = my_raw_utils_test->CcmAdjust(gamma_rgb, ccm);

    Mat bayer = my_raw_utils_test->Bgr2Mosaicking(ccm_rgb);

    Mat rggb_src = my_raw_utils_test->Bayer2Rggb(bayer);

    Mat rggb_wb = my_raw_utils_test->AddWBgain(rggb_src, WbGain);

    Mat gain_rggb = my_raw_utils_test->GainAdjust(rggb_wb, isp_gain);

    Mat rggb_black_sub = my_raw_utils_test->AddBlack(gain_rggb, black_arr);
    resize(rggb_black_sub, rggb_black_sub, rggb_black_sub.size()/2);

    bayer = my_raw_utils_test->Rggb2Bayer(rggb_black_sub);

    return bayer;
}

Mat Raw2Rgb(Mat raw_src) {
    int height  = 640;
    int width   = 437;
    int type    = CV_16UC1;
    int pattern = 1; //BGGR
    int black_arr[4]    = {1024, 1024, 1024, 1024}; //BlackLevel
    int white_arr[4]    = {65535, 65535, 65535, 65535};
    float isp_gain  = 1.37;
    float WbGain[4] = {1.55, 1.0, 1.0, 1.36};
    float ccm[3][3] = {{1.0, 0.0, 0.0},
                       {0.0, 1.0, 0.0},
                       {0.0, 0.0, 1.0}};
    float gamma = 1;

    RawUtils *my_raw_utils_test = new RawUtils();
    //bayer格式转换为rggb格式
    Mat rggb_src = my_raw_utils_test->Bayer2Rggb(raw_src);

    //rggb图像减去黑电平数据
    Mat rggb_black_sub = my_raw_utils_test->SubBlack(rggb_src, black_arr, white_arr);

    //rggb图像作isp gain调整
    Mat gain_rggb = my_raw_utils_test->GainAdjust(rggb_black_sub, isp_gain);

    //rggb图像作白平衡调整
    Mat rggb_wb = my_raw_utils_test->AddWBgain(gain_rggb, WbGain);

    //rggb图像转换回bayer格式
    Mat bayer = my_raw_utils_test->Rggb2Bayer(rggb_wb);

    //bayer图像做demosaicking操作，得到rgb图像
    Mat rgb;
    cvtColor(bayer, rgb, COLOR_BayerRG2BGR_EA);

    //rgb图像作ccm颜色矫正
    Mat ccm_rgb = my_raw_utils_test->CcmAdjust(rgb, ccm);

    //rgb图像作gamma亮度调整
    Mat gamma_rgb = my_raw_utils_test->GammaAdjust(ccm_rgb, gamma);

    return gamma_rgb;
}

int main(int agrc, char* argv[]) {
    Mat src = imread(argv[1]);
    imshow("src.jpg", src);

    src.convertTo(src, CV_16UC3);
    src = src * 256;
    Mat raw = Rgb2Raw(src);
    imshow("raw", raw);

    Mat rgb = Raw2Rgb(raw);
    rgb = rgb / 256;
    rgb.convertTo(rgb, CV_8UC1);

    imshow("dst", rgb);
    
    waitKey(0);
    return 0;
}
