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

class MyWeightPyrNlmBlurTest{
    public:
        MyWeightPyrNlmBlurTest();
        ~MyWeightPyrNlmBlurTest();

		Mat Run(Mat src, float h, int halfKernelSize, int halfSearchSize);

    private:
        Mat Nlm(Mat src, Mat integral_mat, float h, int halfKernelSize, int halfSearchSize);
        Mat GetIntegralImg(Mat src);
};
