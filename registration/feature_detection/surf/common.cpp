#include "common.hpp"

float BoxIntegral(Mat img, int row, int col, int rows, int cols) {
    int r1 = max(0, min(row,          img.rows) - 1);
    int c1 = max(0, min(col,          img.cols) - 1);                                                                                                                                                              
    int r2 = max(0, min(row + rows,   img.rows) - 1);
    int c2 = max(0, min(col + cols,   img.cols) - 1);

    float result = img.at<float>(r1, c1) - img.at<float>(r1, c2) - img.at<float>(r2, c1) + img.at<float>(r2, c2);

    return result;
}

Mat Integral(Mat src) {
    Mat integ_mat = Mat::zeros(src.size(), CV_32FC1);

    float rs = 0.0f;
    for(int j=0; j<src.cols; j++){
        rs += src.at<float>(0, j);
        integ_mat.at<float>(0, j) = rs;
    }

    for(int i=1; i<src.rows; i++) {
        rs = 0.0f;
        for(int j=0; j<src.cols; j++) {
            rs += src.at<float>(i, j);
            integ_mat.at<float>(i, j) = rs + integ_mat.at<float>(i-1, j);
        }
    }
    return integ_mat;
}

