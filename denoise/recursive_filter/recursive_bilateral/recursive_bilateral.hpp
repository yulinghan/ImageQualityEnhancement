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

class MyRecursiveBilateral{
    public:
        MyRecursiveBilateral();
        ~MyRecursiveBilateral();

		Mat Run(Mat src, float sigma_spatial, float sigma_range);

    private:
        Mat BilateralFiltering(Mat src, double *range_table, float sigma_spatial);
};
