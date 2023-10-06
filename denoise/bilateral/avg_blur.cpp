#include "avg_blur.hpp"

MyAvgBlurTest::MyAvgBlurTest() {
}

MyAvgBlurTest::~MyAvgBlurTest() {
}

Mat MyAvgBlurTest::Run(Mat src, int r) {
    Point point(-1, -1);
    int ksize = r*2+1;
    float blur_weight = 1.0/(ksize*ksize);
    Mat Kore = Mat::zeros(ksize, ksize, CV_32FC1);
    Kore.setTo(blur_weight);

    Mat avg_blur;
    filter2D(src, avg_blur, -1, Kore, point, 0, BORDER_CONSTANT);

	return avg_blur;
}
