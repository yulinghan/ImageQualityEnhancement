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

class MyBilateralOriTest{
    public:
        MyBilateralOriTest();
        ~MyBilateralOriTest();

		Mat Run(Mat src, int r, float gauss_sigma, float value_sigma);
};
