#include "light_adjust.hpp"

MyLightAdjustTest::MyLightAdjustTest() {
}

MyLightAdjustTest::~MyLightAdjustTest() {
}

Mat MyLightAdjustTest::Run(Mat src) {
    src.convertTo(src, CV_32FC1, 1/255.0);

    double log_Ave = 0;
    double sum = 0;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            sum += log(0.001 + src.at<float>(i, j));
        }
    }
    log_Ave = exp(sum / (src.rows * src.cols));

    double MaxValue, MinValue;
	minMaxLoc(src, &MinValue, &MaxValue);

    Mat hdr_L(src.rows, src.cols, CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			hdr_L.at<float>(i, j) = log(1 + src.at<float>(i, j) / log_Ave) / log(1 + MaxValue / log_Ave);

			if (src.at<float>(i, j) == 0) {
				hdr_L.at<float>(i, j) = 0;
			} else {
				hdr_L.at<float>(i, j) /= src.at<float>(i, j);
			}
		}
	}
  
    Mat out(src.rows, src.cols, CV_32FC1);
    for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float value = src.at<float>(i, j) *hdr_L.at<float>(i, j);
			out.at<float>(i, j) = value;
		}
	}

    out = out * 255;
    out.convertTo(out, CV_8UC1);

	return out;
}
