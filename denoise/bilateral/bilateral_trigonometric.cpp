#include "bilateral_trigonometric.hpp"

MyBilateralTriTest::MyBilateralTriTest() {
}

MyBilateralTriTest::~MyBilateralTriTest() {
}

int MyBilateralTriTest::calculateCombination(int n, int k) {
    if (k == 0 || k == n) {
        return 1;
    } else {
        return calculateCombination(n - 1, k - 1) + calculateCombination(n - 1, k);
    }
}

Mat MyBilateralTriTest::CalGaussianTemplate(int r, float sigma) {
    float pi = 3.1415926;
    int center = r;
    int ksize = r*2+1;
    float x2, y2;

    Mat Kore = Mat::zeros(Size(ksize, ksize), CV_32FC1);
    for (int i = 0; i < ksize; i++) {
        x2 = pow(i - center, 2);
        for (int j = 0; j < ksize; j++) {
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * pi * sigma;
            Kore.at<float>(i, j) = g;
        }
    }
    return Kore;
}

Mat MyBilateralTriTest::Run(Mat src, int r, float gauss_sigma, float value_sigma) {
    src.convertTo(src, CV_32FC1);
    Mat out = Mat::zeros(src.size(), src.type());

    int T = 255;
    int N = 20;//ceil(0.405 * pow((T/value_sigma), 2));
    float gamma = 1.0 / (sqrt(N) * value_sigma);
    float twoN = pow(2, N);

    float tol = 0.01;
    int M = 0;//ceil(0.5 * (N - sqrt(4*N*log(2/tol))));
    //????

    cout << "M:" << M << ", N:" << N << endl;
    Mat filt = CalGaussianTemplate(r, gauss_sigma);

    Mat out_img1 = Mat::zeros(src.size(), src.type());
    Mat out_img2 = Mat::zeros(src.size(), src.type());

    for(int k=M; k<N-M; k++) {
        float coeff = calculateCombination(N, k) / twoN;
        Mat temp1 = cos(2*k-N) * gamma * src;
        Mat temp2 = sin(2*k-N) * gamma * src;

        Mat phi1, phi2, phi3, phi4;

        Point point(-1, -1);
        filter2D(src.mul(temp1), phi1, -1, filt, point, 0, BORDER_CONSTANT);
        filter2D(src.mul(temp2), phi2, -1, filt, point, 0, BORDER_CONSTANT);
        filter2D(temp1, phi3, -1, filt, point, 0, BORDER_CONSTANT);
        filter2D(temp2, phi4, -1, filt, point, 0, BORDER_CONSTANT);

        out_img1 = out_img1 + coeff * (temp1.mul(phi1) + temp2.mul(phi2));
        out_img2 = out_img2 + coeff * (temp1.mul(phi3) + temp2.mul(phi4));
    }

    out = out_img1 / out_img2;

    out.convertTo(out, CV_8UC1);
    return out;
}
