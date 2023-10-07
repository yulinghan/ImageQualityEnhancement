#include "bilateral_ori.hpp"

MyBilateralOriTest::MyBilateralOriTest() {
}

MyBilateralOriTest::~MyBilateralOriTest() {
}

Mat MyBilateralOriTest::Run(Mat src, int r, float gauss_sigma, float value_sigma) {
    Mat out = Mat::zeros(src.size(), src.type());

    float pi = 3.1415926;
    for (int i=r; i<src.rows-r; i++) {
        for (int j=r; j<src.cols-r; j++) {
            float cur_weight = 0.0;
            float value = 0.0;
            for (int m=-r; m<=r; m++){
				for (int n=-r; n<=r; n++){
                    float weight_gauss = exp(-(m*m + n*n) / (2 * gauss_sigma * gauss_sigma));
                    weight_gauss /= 2 * pi * gauss_sigma;

                    int cur_diff = abs(src.at<uchar>(i,j) - src.at<uchar>(i+m,j+n));
                    float weight_value = exp(-(cur_diff*cur_diff) / (2 * value_sigma * value_sigma));
                    float weight = weight_gauss * weight_value;
                    value += src.at<uchar>(i+m,j+n) * weight;
                    cur_weight += weight;
                }
            }
            value = value / cur_weight;
            out.at<uchar>(i, j) = value;
        }
    }
    return out;
}
