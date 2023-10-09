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

class MyRecursiveGaussian{
    public:
        MyRecursiveGaussian();
        ~MyRecursiveGaussian();

		Mat Run(Mat src, int r);

    private:
        void CalcGaussCof(float Radius, float &B0, float &B1, float &B2, float &B3);
        Mat GaussBlurFromLeftToRight(Mat src, float B0, float B1, float B2, float B3);
};
