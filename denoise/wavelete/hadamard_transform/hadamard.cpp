#include "hadamard.hpp"

MyHadamardTest::MyHadamardTest() {
}

MyHadamardTest::~MyHadamardTest() {
}

void MyHadamardTest::hadamard_transform(Mat &src, int N, int D, Mat &out){
    if (N == 1)
        return;
    else if (N == 2) {
        const float a = src.at<float>(0, D+0);
        const float b = src.at<float>(0, D+1);
        src.at<float>(0, D+0) = a + b;
        src.at<float>(0, D+1) = a - b;
    } else {
        const unsigned n = N / 2;
        for (unsigned k = 0; k < n; k++) {
            const float a = src.at<float>(0, D+2*k);
            const float b = src.at<float>(0, D+2*k+1);
            src.at<float>(0, D+k) = a + b;
            out.at<float>(0, k) = a - b;
        }

        for (unsigned k = 0; k < n; k++) {
            src.at<float>(0, D+n+k) = out.at<float>(0, k);
        }
        hadamard_transform(src, n, D, out);
        hadamard_transform(src, n, D + n, out);
    }
}
