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
#include "contrast_adjust.hpp"
#include "light_adjust.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1]);

    vector<Mat> channels;
    split(src, channels);

    //自适应亮度调整
    for(int i=0; i<channels.size(); i++) {
        MyLightAdjustTest *my_light_adjust_test = new MyLightAdjustTest();
        channels[i] = my_light_adjust_test->Run(channels[i]);
    }

    Mat light_mat;
    merge(channels, light_mat);
    imwrite(argv[2], light_mat);

    //自适应对比度调整
    for(int i=0; i<channels.size(); i++) {
        MyContrastAdjustTest *my_contrast_adjust_test = new MyContrastAdjustTest();
        channels[i] = my_contrast_adjust_test->Run(channels[i], 35);
    }

    Mat out;
    merge(channels, out);

    imwrite(argv[3], out);
    return 0;
}
