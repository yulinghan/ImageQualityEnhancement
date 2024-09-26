#include "multiscalesharpen.hpp"

MyMultiScaleSharpenTest::MyMultiScaleSharpenTest() {
}

MyMultiScaleSharpenTest::~MyMultiScaleSharpenTest() {
}

int MyMultiScaleSharpenTest::Sign(int x) {
    return (x >> 31) | (unsigned(-x)) >> 31;        
}

Mat MyMultiScaleSharpenTest::Run(Mat src, int r, float scale) {
    Mat out = Mat::zeros(src.size(), src.type());

    Mat gauss_mat1, gauss_mat2, gauss_mat3;
    GaussianBlur(src, gauss_mat1, Size(r, r), 0, 0);
    GaussianBlur(src, gauss_mat2, Size(r*2+1, r*2+1), 0, 0);
    GaussianBlur(src, gauss_mat3, Size(r*4+1, r*4+1), 0, 0);

    for(int i=0; i<gauss_mat1.rows; i++) {
        uchar *ptr_src    = src.ptr(i);
        uchar *ptr_gauss1 = gauss_mat1.ptr(i);
        uchar *ptr_gauss2 = gauss_mat2.ptr(i);
        uchar *ptr_gauss3 = gauss_mat3.ptr(i);
        uchar *ptr_out    = out.ptr(i);
        for(int j=0; j<gauss_mat1.cols; j++) {
            int value1 = ptr_src[j]    - ptr_gauss1[j];
            int value2 = ptr_gauss1[j] - ptr_gauss2[j];
            int value3 = ptr_gauss2[j] - ptr_gauss3[j];
            
            int value = ((4-2*Sign(value1))*value1 + 2*value2 + value3)/4 + ptr_src[j];
            ptr_out[j] = max(min(value, 255), 0);
        }
    }

	return out;
}
