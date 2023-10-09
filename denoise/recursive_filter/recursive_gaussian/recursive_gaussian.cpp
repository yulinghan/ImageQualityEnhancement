#include "recursive_gaussian.hpp"

MyRecursiveGaussian::MyRecursiveGaussian() {
}

MyRecursiveGaussian::~MyRecursiveGaussian() {
}

void MyRecursiveGaussian::CalcGaussCof(float Radius, float &B0, float &B1, float &B2, float &B3) {
    float Q, B;
    if (Radius >= 2.5)
        Q = (0.98711 * Radius - 0.96330);
    else if ((Radius >= 0.5) && (Radius < 2.5))
        Q = (3.97156 - 4.14554 * sqrt(1 - 0.26891 * Radius));
    else
        Q = 0.1147705018520355224609375;

    B = 1.57825 + 2.44413 * Q + 1.4281 * Q * Q + 0.422205 * Q * Q * Q;
    B1 = 2.44413 * Q + 2.85619 * Q * Q + 1.26661 * Q * Q * Q;
    B2 = -1.4281 * Q * Q - 1.26661 * Q * Q * Q;
    B3 = 0.422205 * Q * Q * Q;

    B0 = 1.0 - (B1 + B2 + B3) / B;
    B1 = B1 / B;
    B2 = B2 / B;
    B3 = B3 / B;
}

Mat MyRecursiveGaussian::GaussBlurFromLeftToRight(Mat src, float B0, float B1, float B2, float B3) {
    Mat out = Mat::zeros(src.size(), CV_8UC1);

    for(int i=0; i<src.rows; i++) {
        Mat w1  = Mat::zeros(Size(src.cols+3, 1), CV_32FC1);
        Mat w2  = Mat::zeros(Size(src.cols+3, 1), CV_32FC1);
        float *ptr_w1  = w1.ptr<float>(0);
        float *ptr_w2  = w2.ptr<float>(0);
        uchar *ptr_src = src.ptr<uchar>(i);
        uchar *ptr_out = out.ptr<uchar>(i);

        ptr_w1[0] = ptr_w1[1] = ptr_w1[2] = ptr_src[0];
        for(int n=3, j=0; j<src.cols; n++, j++) {
            ptr_w1[n] = B0*ptr_src[j] + B1*ptr_w1[n-1] + B2*ptr_w1[n-2] + B3*ptr_w1[n-3];
        }

        ptr_w2[src.cols] = ptr_w2[src.cols+1] = ptr_w2[src.cols+2] = ptr_w1[src.cols+2];
        for(int j=src.cols-1; j>=0; j--) {
            ptr_out[j] = ptr_w2[j] = B0*ptr_w1[j+3] + B1*ptr_w2[j+1] + B2*ptr_w2[j+2] + B3*ptr_w2[j+3];
        }
    }

    for(int i=0; i<src.cols; i++) {
        Mat w1  = Mat::zeros(Size(src.rows+3, 1), CV_32FC1);
        Mat w2  = Mat::zeros(Size(src.rows+3, 1), CV_32FC1);
        float *ptr_w1  = w1.ptr<float>(0);
        float *ptr_w2  = w2.ptr<float>(0);

        ptr_w1[0] = ptr_w1[1] = ptr_w1[2] = src.at<uchar>(0, i);
        for(int n=3, j=0; j<src.rows; n++, j++) {
            ptr_w1[n] = B0*out.at<uchar>(j, i) + B1*ptr_w1[n-1] + B2*ptr_w1[n-2] + B3*ptr_w1[n-3];
        }

        ptr_w2[src.rows] = ptr_w2[src.rows+1] = ptr_w2[src.rows+2] = ptr_w1[src.rows+2];
        for(int j=src.rows-1; j>=0; j--) {
           out.at<uchar>(j, i) = ptr_w2[j] = B0*ptr_w1[j+3] + B1*ptr_w2[j+1] + B2*ptr_w2[j+2] + B3*ptr_w2[j+3];
        }
    }

    return out;   
}

Mat MyRecursiveGaussian::Run(Mat src, int r) {
    float B0, B1, B2, B3;
    CalcGaussCof(r, B0, B1, B2, B3);
    cout << B0 << ", " << B1 << ", B2:" << B2 << ", B3:" << B3 << endl;
    Mat out = GaussBlurFromLeftToRight(src, B0, B1, B2, B3);
    

	return out;
}
