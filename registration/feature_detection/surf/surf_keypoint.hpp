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
#include "common.hpp"                                                                                                                                                                                                           

class MySurfKeyPointTest{
    public:
        MySurfKeyPointTest();
        ~MySurfKeyPointTest();

        vector<MyKeyPoint> run(Mat input);
        Mat DispKeyPoint(Mat src, vector<MyKeyPoint> key_point_vec);

    private:
        vector<vector<ResponseLayer>> buildResponseMap(Mat integ_mat);
    
        bool isExtremum(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b);
        void interpolateStep(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b, 
                                  double &xi, double &xr, double &xc);
        bool interpolateExtremum(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b, MyKeyPoint &key_point);
        Mat hessian3D(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b);
        Mat deriv3D(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b);
        vector<MyKeyPoint> GetKeyPoints(vector<vector<ResponseLayer>> response_layer);


    private:
        int OCTAVES = 5;//组数默认设置为5
        int INTERVALS = 4;//层数默认设置为4
        float THRES = 0.0004f;//斑点响应阈值默认设置为0.0004
        int INIT_SAMPLE = 2;//采样间隔为2
};
