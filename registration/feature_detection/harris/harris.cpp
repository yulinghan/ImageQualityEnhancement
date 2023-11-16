#include "harris.hpp"

MyHarrisTest::MyHarrisTest() {
}

MyHarrisTest::~MyHarrisTest() {
}

void MyHarrisTest::GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy){
    Mat sobelx = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobely = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    Mat Ix, Iy;
    Point point(-1, -1);
    filter2D(src, Ix, -1, sobelx, point, 0, BORDER_CONSTANT);
    filter2D(src, Iy, -1, sobely, point, 0, BORDER_CONSTANT);

    Ixx = Ix.mul(Ix);
    Iyy = Iy.mul(Iy);
    Ixy = Ix.mul(Iy);
}

Mat MyHarrisTest::ScoreImg(Mat Ixx, Mat Ixy, Mat Iyy) {
    Mat res = Mat::zeros(Ixx.size(),CV_32FC1);

    for(int r = 0; r < Ixx.rows; r++) {
        for(int c = 0; c < Ixx.cols; c++) {
            Mat M = (Mat_<float>(2,2)<<Ixx.at<float>(r,c),Ixy.at<float>(r,c),Ixy.at<float>(r,c),Iyy.at<float>(r,c));
            //determinant:计算矩阵行列式
            //trace:计算矩阵对角线之和
            float score = determinant(M) - 0.05*trace(M)[0]*trace(M)[0];
            res.at<float>(r,c) = score;
        }
    }
    return res;
}

vector<Point> MyHarrisTest::get_corners(Mat res, float thresh) {
    vector<Point> corners;
    for (int r=0; r<res.rows; r++) {
        for(int c=0; c<res.cols; c++) { 
            if(res.at<float>(r,c)>thresh) {
                corners.emplace_back(c,r);
            }
        }
    }

    return corners;
}

Mat MyHarrisTest::CornersShow(Mat src, vector<Point> corners) {
    Mat dst = src.clone();

    for(int i=0; i<corners.size(); i++) {
        circle(dst, corners[i], 3, Scalar(255, 0, 0), -1);  // 画半径为1的圆(画点）
    }

    return dst;
}

vector<cv::Point> MyHarrisTest::run(Mat src) {
    src.convertTo(src, CV_32FC1, 1/255.0);

    Mat Ixx, Ixy, Iyy;
    GetGradient(src, Ixx, Ixy, Iyy);

    int kern_size = 5;
    GaussianBlur(Ixx, Ixx, Size(kern_size, kern_size), 0, 0);
    GaussianBlur(Ixy, Ixy, Size(kern_size, kern_size), 0, 0);
    GaussianBlur(Iyy, Iyy, Size(kern_size, kern_size), 0, 0);

    Mat res = ScoreImg(Ixx, Ixy, Iyy);
    float thresh = abs(mean(src)[0]);
    vector<Point> corners = get_corners(res, thresh);

    return corners;
}
