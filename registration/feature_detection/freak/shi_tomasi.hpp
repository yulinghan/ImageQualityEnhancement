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

struct greaterThanPtr {
    bool operator () (const float * a, const float * b) const
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

class MyShiTomasiTest{
    public:
        MyShiTomasiTest();
        ~MyShiTomasiTest();

        vector<Point> run(Mat input);
        Mat CornersShow(Mat src, vector<Point> corners);

    private:
        vector<Point> get_corners(Mat res, float thresh);
        void GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy);
        Mat CalcMinEigenVal(Mat Ixx, Mat Ixy, Mat Iyy);

        vector<float*> CornersChoice(Mat eig, float thresh);

        void DistanceChoice(Mat eig, vector<float*> tmpCorners, float minDistance, int maxCorners,
            vector<Point> &corners, vector<float> &scores);
};
