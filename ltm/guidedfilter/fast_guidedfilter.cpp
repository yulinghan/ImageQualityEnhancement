#include "fast_guidedfilter.hpp"

MyFastGuidedfilterTest::MyFastGuidedfilterTest() {
}

MyFastGuidedfilterTest::~MyFastGuidedfilterTest() {
}

Mat MyFastGuidedfilterTest::Run(Mat I, Mat p, int r, float eps, int size) {
    r = r / size;
    int wsize = 2 * r + 1;
    
    Mat small_I, small_p;
    resize(I, small_I, I.size()/size, 0, 0, INTER_AREA);
    resize(p, small_p, p.size()/size, 0, 0, INTER_AREA);

    Mat mean_I;
    boxFilter(small_I, mean_I, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat mean_p;
    boxFilter(small_p, mean_p, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat mean_II;
    mean_II = small_I.mul(small_I);
    boxFilter(mean_II, mean_II, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat mean_Ip;
    mean_Ip = small_I.mul(small_p);
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

    resize(mean_a, mean_a, I.size());
    resize(mean_b, mean_b, I.size());

    Mat out = mean_a.mul(I) + mean_b;

    return out;
}
