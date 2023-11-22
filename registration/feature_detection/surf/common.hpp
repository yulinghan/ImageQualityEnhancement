#ifndef _COMMON_H_
#define _COMMON_H_

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

struct ResponseLayer{
    int width;     //当前层的width
    int height;    //当前层的height
    int step;      //当前层相对原图缩放倍数
    int filter;    //当前层boxfilter滤波半径
    Mat responses;
    Mat laplacian;
};

struct MyKeyPoint{
    float x;              //极值点x坐标
    float y;              //极值点y坐标
    float scale;          //极值点所在尺度
    float laplacian;      //???
    float orientation;    //以x轴正方向，绕逆时针旋转的角度
    float descriptor[64]; //64维特征向量
};

float BoxIntegral(Mat img, int row, int col, int rows, int cols);
Mat Integral(Mat src);

#endif
