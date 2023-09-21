#include "contrast_adjust.hpp"

MyContrastAdjustTest::MyContrastAdjustTest() {
}

MyContrastAdjustTest::~MyContrastAdjustTest() {
}

double MyContrastAdjustTest::ContrastStrength(Mat src) {
    Scalar mean;
    Scalar stddev;

    meanStdDev(src, mean, stddev);

    double p = 1.0;
    if(stddev.val[0] <= 3){
        p = 3.0;
    } else if(stddev.val[0] <= 10) {
        p = (27 - 2 * stddev.val[0]) / 7;
    } else {
        p = 1.0;
    }
    
    return p;
}

Mat MyContrastAdjustTest::ContrastAdjust(Mat src, double c_strength, int r) {
    src.convertTo(src, CV_32FC1, 1/255.0);

    Mat gauss_mat;
    GaussianBlur(src, gauss_mat, Size(r, r), 0, 0);

    Mat div_mat = (gauss_mat + 0.001) / (src + 0.001);

    Mat E;
    pow(div_mat, c_strength, E);

    Mat S = Mat::zeros(E.size(), E.type());
    for(int i=0; i<S.rows; i++) {
        for(int j=0; j<S.cols; j++) {
            S.at<float>(i, j) = pow(src.at<float>(i, j), E.at<float>(i, j));
        }
    }

    Mat dst = S * 255;
    dst.convertTo(dst, CV_8UC1);

    return dst;
}


Mat MyContrastAdjustTest::Run(Mat src, int r) {
    double c_strength = ContrastStrength(src);
    Mat out = ContrastAdjust(src, c_strength, r);

	return out;
}
