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

class MyTonemapMantiuk{
    public:
        MyTonemapMantiuk();
        ~MyTonemapMantiuk();

		Mat Run(Mat src, float gamma, float power, float saturation);
    
    private:
        void getContrast(Mat src, std::vector<Mat>& x_contrast, std::vector<Mat>& y_contrast);
        void getGradient(Mat src, Mat& dst, int pos);
        void log_(const Mat& src, Mat& dst);
        void signedPow(Mat src, float power, Mat& dst);
        void mapContrast(Mat& contrast);
        void calculateSum(std::vector<Mat>& x_contrast, std::vector<Mat>& y_contrast, Mat& sum);
        void calculateProduct(Mat src, Mat& dst);
        Mat mapLuminance(Mat src, Mat lum, Mat new_lum, float saturation);
        Mat linear(Mat src, float gamma);

    private:
        float scale;
};
