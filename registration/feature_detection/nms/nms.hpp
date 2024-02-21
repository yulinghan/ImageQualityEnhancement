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

class MyNmsTest{
    public:
        MyNmsTest();
        ~MyNmsTest();

        vector<KeyPoint> run(Mat input, int maxCorners);
        Mat CornersShow(Mat src, vector<KeyPoint> corners);
        void DistanceChoice(float minDistance, int maxCorners, vector<KeyPoint> &corners);
        vector<KeyPoint> ANMS(vector<KeyPoint> kpts, int maxCorners);
        vector<KeyPoint> ssc(vector<KeyPoint> keyPoints, int numRetPoints, float tolerance, int cols, int rows);

    private:
        void GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy);
        Mat CalcMinEigenVal(Mat Ixx, Mat Ixy, Mat Iyy);

        vector<float*> CornersChoice(Mat eig, float thresh);
        void GetKeyPoint(Mat eig, vector<float*> tmpCorners, int maxCorners, vector<KeyPoint> &corners);
        
        template <typename T> vector<size_t> sort_indexes(const vector<T> &v);
        double computeR(Point2i x1, Point2i x2);

    private:
        Mat eig_;
        vector<float*> tmpCorners_;

};
