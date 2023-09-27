#include "weight_guidedfilter.hpp"

MyWeightGuidedfilterTest::MyWeightGuidedfilterTest() {
}

MyWeightGuidedfilterTest::~MyWeightGuidedfilterTest() {
}

Mat MyWeightGuidedfilterTest::EdgeAwareWeighting(Mat I, Mat var_I){
	double MinVal = 0, MaxVal = 0;
	Point minPoint, maxPoint;
	minMaxLoc(I, &MinVal, &MaxVal, NULL, NULL);
	float L = MaxVal - MinVal;

	float eps = (0.001*L) *(0.001*L);
	Mat var_I1 = var_I + eps;
	Mat gamma = var_I1*cv::sum(1 / var_I1)[0] / I.total();
	GaussianBlur(gamma, gamma, cv::Size(3, 3), 2);

    return gamma;
}

Mat MyWeightGuidedfilterTest::Run(Mat I, Mat p, int r, float eps) {
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

    Mat gamma = EdgeAwareWeighting(I, var_I);

    Mat a, b;
    divide(cov_Ip, (var_I + eps / gamma), a);
    subtract(mean_p, a.mul(mean_I), b);

    Mat mean_a, mean_b;
    boxFilter(a, mean_a, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);
    boxFilter(b, mean_b, -1, Size(wsize, wsize), Point(-1, -1), true, BORDER_REFLECT);

    Mat out = mean_a.mul(I) + mean_b;

    return out;
}
