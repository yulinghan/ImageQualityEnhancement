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
#include "pyr_contrast.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1]);

    vector<Mat> channels;
    split(src, channels);

    for(int  i=0; i<channels.size(); i++) {
    	MyPyrContrastTest *my_pyr_contrast_test = new MyPyrContrastTest();
        channels[i] = my_pyr_contrast_test->Run(channels[i]);
    }

    Mat out;
    merge(channels, out);
    imwrite(argv[2], out);

    return 0;
}
