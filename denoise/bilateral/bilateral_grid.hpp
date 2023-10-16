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

class MyBilateralGridTest{
    public:
        MyBilateralGridTest();
        ~MyBilateralGridTest();

		Mat Run(Mat src, int r, float gauss_sigma, float value_sigma);

    private:
        vector<vector<vector<float>>> Cal3DGaussianTemplate(int r, float gauss_sigma, float value_sigma);
        int *** qx_alloci(int n,int r,int c);
};
