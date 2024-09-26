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
#include "musica.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1]);
    float power = 0.7;

    vector<Mat> channels;
    split(src, channels);

    for(int  i=0; i<channels.size(); i++) {
    	MyMusicaTest *my_musica_test = new MyMusicaTest();
        channels[i] = my_musica_test->Run(channels[i], power);
    }

    Mat out;
    merge(channels, out);
    imwrite(argv[2], out);

    return 0;
}
