#include "alpha.hpp"

MyAlphaTest::MyAlphaTest() {
}

MyAlphaTest::~MyAlphaTest() {
}

Mat MyAlphaTest::Run(Mat src1, Mat src2, Mat mask) {
	Mat out = Mat::zeros(src1.size(), src1.type());
	int channels = src1.channels();

	for(int i=0; i<src1.rows; i++) {
		uchar *ptr_src1 = src1.ptr(i);
		uchar *ptr_src2 = src2.ptr(i);
		uchar *ptr_mask = mask.ptr(i);
		uchar *ptr_out  = out.ptr(i);
		for(int j=0; j<src1.cols; j++) {
			for(int c=0; c<channels; c++) {
				int value = ptr_src1[j*channels + c] * ptr_mask[j] + 
										+ ptr_src2[j*channels + c] * (255 - ptr_mask[j]);
				ptr_out[j*channels + c] = value / 255;

			}
		}
	}
	return out;
}
