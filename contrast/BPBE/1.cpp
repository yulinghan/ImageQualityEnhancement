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
#include "bpbe.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1]);
    imshow("src", src);

    src.convertTo(src, CV_32F);
    vector<Mat> channels;
    split(src, channels);
    Mat src_y = (channels[0] + channels[1] + channels[2]) / 3;

    MyBPBETest *my_bpbe_test = new MyBPBETest();
    Mat weight = my_bpbe_test->Run(src_y);

    for(int i=0; i<channels.size(); i++) {
        channels[i] = channels[i].mul(weight);
    }

    Mat out;
    merge(channels, out);
    imshow("dst", out/255);

    waitKey(0);

    return 0;
}
