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

class MyHaarFusionTest{
    public:
        MyHaarFusionTest();
        ~MyHaarFusionTest();

		Mat Run(Mat src);

    private:
        void harr_wave_decompose(Mat &src, Mat &LL, Mat &LH, Mat &HL, Mat &HH);
        Mat harr_wave_recover(Mat &LL, Mat &LH, Mat &HL, Mat &HH);
};
