#include "usm.hpp"

MyUsmTest::MyUsmTest() {
}

MyUsmTest::~MyUsmTest() {
}

Mat MyUsmTest::Run(Mat src, int r, int Thr, float scale) {
    Mat out = Mat::zeros(src.size(), src.type());

    Mat gauss_mat;
    GaussianBlur(src, gauss_mat, Size(r, r), 0, 0);
    for(int i=0; i<gauss_mat.rows; i++) {
        uchar *ptr_src   = src.ptr(i);
        uchar *ptr_gauss = gauss_mat.ptr(i);
        uchar *ptr_out   = out.ptr(i);
        for(int j=0; j<gauss_mat.cols; j++) {
            int value = ptr_src[j] - ptr_gauss[j];
            if (abs(value) > Thr) {
                value = ptr_src[j] + scale*value;
            } else {
                value = ptr_src[j];
            }
            ptr_out[j] = max(min(value, 255), 0);
        }
    }

	return out;
}
