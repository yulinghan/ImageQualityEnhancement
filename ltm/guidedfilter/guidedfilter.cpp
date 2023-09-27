#include "guidedfilter.hpp"

MyGuidedfilterTest::MyGuidedfilterTest() {
}

MyGuidedfilterTest::~MyGuidedfilterTest() {
}

Mat MyGuidedfilterTest::Run(Mat I, Mat p, int r, float eps) {
    int wsize = 2 * r + 1;

    Mat mean_I;
    boxFilter(I, mean_I, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat mean_p;
    boxFilter(p, mean_p, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat mean_II;
    mean_II = I.mul(I);
    boxFilter(mean_II, mean_II, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat mean_Ip;
    mean_Ip = I.mul(p);
    boxFilter(mean_Ip, mean_Ip, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat var_I, mean_mul_I;
    mean_mul_I=mean_I.mul(mean_I);
    subtract(mean_II, mean_mul_I, var_I);

    Mat cov_Ip;
    subtract(mean_Ip, mean_I.mul(mean_p), cov_Ip);

    Mat a, b;
    divide(cov_Ip, (var_I+eps),a);
    subtract(mean_p, a.mul(mean_I), b);

    Mat mean_a, mean_b;
    boxFilter(a, mean_a, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);
    boxFilter(b, mean_b, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat out = mean_a.mul(I) + mean_b;

    return out;
}
