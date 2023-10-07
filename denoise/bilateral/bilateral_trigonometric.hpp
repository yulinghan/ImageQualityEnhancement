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

class MyBilateralTriTest{
    public:
        MyBilateralTriTest();
        ~MyBilateralTriTest();

		Mat Run(Mat src, int r, float gauss_sigma, float value_sigma);

    private:
        int calculateCombination(int n, int k);
        Mat CalGaussianTemplate(int r, float sigma);
};
