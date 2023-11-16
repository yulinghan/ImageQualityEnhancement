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

class MyHarrisTest{
    public:
        MyHarrisTest();
        ~MyHarrisTest();

        vector<Point> run(Mat input);
        Mat CornersShow(Mat src, vector<Point> corners);

    private:
        vector<Point> get_corners(Mat res, float thresh);
        void GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy);
        Mat ScoreImg(Mat Ixx, Mat Ixy, Mat Iyy);
};
