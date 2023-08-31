#ifndef _MTBFramesRegistration__H_
#define _MTBFramesRegistration__H_

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
class MyMtbTest{
    public:
        MyMtbTest();
        ~MyMtbTest();

        Mat SparseBlur(Mat src, int sparseFactor);
        void Downsample(Mat& src, Mat& dst);
        void BuildPyr(const Mat& img, std::vector<Mat>& pyr);
        int GetAverage(Mat& img);
        int GetMedian(Mat& img);
        void ComputeBitmaps(InputArray _img, OutputArray _tb);
        Mat ShiftMat(Mat src, Point shift);
        Point CalculateShift(const Mat& img, const vector<Mat> tbRef, const Rect& ROI);
        Mat RegistrationY(Mat image, vector<Mat> tbRef, Rect ROI);
        Mat Run(Mat input, Mat ref);

    private:
        int max_level_ = 5;
        int roi_length_ = 1024;
        Point shift_;
};
#endif
