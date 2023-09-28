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

class MyBilateralBlurTest{
    public:
        MyBilateralBlurTest();
        ~MyBilateralBlurTest();

		Mat Run(Mat src, int r, float gauss_sigma, float value_sigma);

    private:
        Mat CalGaussianTemplate(int ksize, float sigma);
        vector<float> CalValueTemplate(float sigma);
        Mat BilateralBlur(Mat src, Mat gaussian_kore, vector<float> val_weight_arr, int r);
};
