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

class MyIntegralNlmBlurTest{
    public:
        MyIntegralNlmBlurTest();
        ~MyIntegralNlmBlurTest();

		Mat Run(Mat src, float h, int halfKernelSize, int halfSearchSize);

    private:
        Mat GetIntegralImg(Mat src, int halfSearchSize, int t1, int t2);
        Mat Nlm(Mat src, float h, int halfKernelSize, int halfSearchSize);
};
