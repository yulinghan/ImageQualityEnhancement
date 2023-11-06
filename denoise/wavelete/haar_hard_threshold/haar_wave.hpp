#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>

using namespace cv;
using namespace std;

class MyHaarWaveTest{
    public:
        MyHaarWaveTest();
        ~MyHaarWaveTest();

		Mat harr_wave_decompose(Mat src, int level);
        Mat harr_wave_denoise(Mat src, int threshold, int level);
        Mat harr_wave_recover(Mat src_img, int level);
};
