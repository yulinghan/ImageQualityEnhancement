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
#include <omp.h>

using namespace cv;
using namespace std;

class MyContrastAdjustTest{
    public:
        MyContrastAdjustTest();
        ~MyContrastAdjustTest();

		Mat Run(Mat src, int r);

    private:
        double ContrastStrength(Mat src);
        Mat ContrastAdjust(Mat src, double c_strength, int r);
};
