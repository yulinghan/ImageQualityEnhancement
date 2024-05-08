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

class MyTonemapReinhard{
    public:
        MyTonemapReinhard();
        ~MyTonemapReinhard();

		Mat Run(Mat src, float gamma, float intensity, 
                            float light_adapt, float color_adapt);

    private:
        Mat linear(Mat src, float gamma);
        void log_(const Mat& src, Mat& dst);
};
