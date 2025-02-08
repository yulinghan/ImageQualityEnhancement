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

class MySteerableFusionTest{
    public:
        MySteerableFusionTest();
        ~MySteerableFusionTest();

		Mat Run(Mat src);

    private:
        void Shift(Mat &src);
        Mat myfft(Mat &src);
        void GetFFTFreq(Mat complexI, int r, Mat &fft_low, vector<Mat> &fft_freq_arr);
        Mat myifft(Mat src);
};
