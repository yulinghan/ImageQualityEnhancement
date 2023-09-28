#include "gaussian_blur.hpp"

MyGaussianBlurTest::MyGaussianBlurTest() {
}

MyGaussianBlurTest::~MyGaussianBlurTest() {
}

Mat MyGaussianBlurTest::CalGaussianTemplate(int r, float sigma) {
    float pi = 3.1415926;
    int center = r;
    int ksize = r*2 + 1;
    float x2, y2;

    float k = 0;
    Mat Kore = Mat::zeros(Size(ksize, ksize), CV_32FC1);
    for (int i = 0; i < ksize; i++) {
        x2 = pow(i - center, 2);
        for (int j = 0; j < ksize; j++) {
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * pi * sigma;
            Kore.at<float>(i, j) = g;
            k += g;
        }
    }

    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            Kore.at<float>(i, j) /= k;
        }
    }
    return Kore;
}

Mat MyGaussianBlurTest::Run(Mat src, int r, float sigma) {
    Mat Kore = CalGaussianTemplate(r, sigma);
    Mat gaussian_blur;
    Point point(-1, -1);

    filter2D(src, gaussian_blur, -1, Kore, point, 0, BORDER_CONSTANT);

	return gaussian_blur;
}
