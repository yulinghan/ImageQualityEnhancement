#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>  
#include <unistd.h>

using namespace std;
using namespace cv;

class MyAkazeKeyPointsTest{
    public:
        MyAkazeKeyPointsTest();
        ~MyAkazeKeyPointsTest();

        vector<KeyPoint> run(vector<vector<Mat>> scale_space_arr);
        Mat DispKeyPoint(Mat src, vector<KeyPoint> key_point_vec);

    private:
        bool isExtremum(int r, int c, Mat t, Mat m, Mat b);
        void interpolateStep(int r, int c, Mat t, Mat m, Mat b, double &xi, double &xr, double &xc);
        bool interpolateExtremum(int r, int c, Mat t, Mat m, Mat b, KeyPoint &key_point, int octave, int class_id);
        Mat hessian3D(int r, int c, Mat t, Mat m, Mat b);
        Mat deriv3D(int r, int c, Mat t, Mat m, Mat b);
        vector<KeyPoint> GetKeyPoints(vector<vector<Mat>> response_layer);

        void GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy);
        Mat ScoreImg(Mat Ixx, Mat Ixy, Mat Iyy);
        vector<vector<Mat>> DeterminantHessianResponse(vector<vector<Mat>> scale_space_arr);

    private:
        float THRES = 0.0004f;//斑点响应阈值默认设置为0.0004
};
