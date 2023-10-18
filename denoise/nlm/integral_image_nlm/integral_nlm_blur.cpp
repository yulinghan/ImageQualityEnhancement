#include "integral_nlm_blur.hpp"

MyIntegralNlmBlurTest::MyIntegralNlmBlurTest() {
}

MyIntegralNlmBlurTest::~MyIntegralNlmBlurTest() {
}

Mat MyIntegralNlmBlurTest::GetIntegralImg(Mat src, int halfSearchSize, int t1, int t2) {
    src.convertTo(src, CV_32F);
    Mat d1 = src(Range(halfSearchSize, src.rows-halfSearchSize), Range(halfSearchSize, src.cols-halfSearchSize));
    Mat d2 = src(Range(halfSearchSize+t1, src.rows-halfSearchSize+t1), Range(halfSearchSize+t2, src.cols-halfSearchSize+t2));

    Mat out = (d1-d2).mul(d1-d2);

    for (int i=0;i<out.rows;i++) {
        for (int j = 1; j<out.cols; j++) {
            out.at<float>(i, j) += out.at<float>(i, j - 1);
        }
    }

    for (int i = 1; i<out.rows; i++) {
        for (int j = 0; j<out.cols; j++) {
            out.at<float>(i, j) += out.at<float>(i-1, j);
        }
    }

    return out;
}

Mat MyIntegralNlmBlurTest::Nlm(Mat src, float h, int hk, int hs) {
    Mat dst    = Mat::zeros(src.size(), CV_32FC1);
    Mat weight = Mat::zeros(src.size(), CV_32FC1);
    int boardSize = hk + hs + 1;
    Mat boardSrc;
    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展

    float h1 = 1.0 / (h*h);
    float h2 = 1.0 / (2*hk+1) / (2*hk+1);
    h = h1*h2;

    int rows = src.rows;
    int cols = src.cols;

    for(int t1 = -hs; t1<hs; t1++) {
        for(int t2 = -hs; t2<hs; t2++) {
            Mat integral_mat = GetIntegralImg(boardSrc, hs, t1, t2);
            for(int i=0; i<src.rows; i++) {
                for(int j=0; j<src.cols; j++) {
                    int i1 = i+hk+1;
                    int j1 = j+hk+1;
                    float sum = integral_mat.at<float>(i1+hk, j1+hk) + integral_mat.at<float>(i1-hk-1, j1-hk-1) 
                                - integral_mat.at<float>(i1+hk, j1-hk-1) - integral_mat.at<float>(i1-hk-1, j1+hk);
                    float cur_weight = exp(-sum*h);
                    weight.at<float>(i, j) += cur_weight;
                    dst.at<float>(i, j) += boardSrc.at<uchar>(hk+hs+t1+i, hk+hs+t2+j) * cur_weight;
                }
            }
        }
    }

    dst = dst / weight;
    dst.convertTo(dst, CV_8UC1);   

    return dst;
}

Mat MyIntegralNlmBlurTest::Run(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat nlm_blur = Nlm(src, h, halfKernelSize, halfSearchSize);

	return nlm_blur;
}
