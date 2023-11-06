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
#include "haar_wave.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    int level = 3;
	MyHaarWaveTest *my_haar_wave_test = new MyHaarWaveTest();
    Mat haar_wave = my_haar_wave_test->harr_wave_decompose(src, level);
    imshow("haar_wave", haar_wave/255);

    int threshold = 5;
    Mat denoise_haar_wave = my_haar_wave_test->harr_wave_denoise(haar_wave, threshold, level);

    Mat harr_wave_recover = my_haar_wave_test->harr_wave_recover(denoise_haar_wave, level);
    imshow("harr_wave_recover", harr_wave_recover);

    waitKey(0);

    return 0;
}
