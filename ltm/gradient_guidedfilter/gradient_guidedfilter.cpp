#include "gradient_guidedfilter.hpp"

MyGradientGuidedfilterTest::MyGradientGuidedfilterTest() {
}

MyGradientGuidedfilterTest::~MyGradientGuidedfilterTest() {
}

Mat MyGradientGuidedfilterTest::Run(Mat I, Mat p, int r, float eps) {
    Mat n_mat = Mat::ones(I.size(), I.type());
    boxFilter(n_mat, n_mat, -1, Size(r, r));

    Mat mean_I, mean_p, mean_Ip, cov_Ip, mean_II, var_I;
    boxFilter(I, mean_I, -1, Size(r, r));
    mean_I = mean_I / n_mat;

    boxFilter(p, mean_p, -1, Size(r, r));
    mean_p = mean_p / n_mat;

    boxFilter(I.mul(p), mean_Ip, -1, Size(r, r));
    mean_Ip = mean_Ip / n_mat;

    cov_Ip  = mean_Ip - mean_I.mul(mean_p);

    boxFilter(I.mul(I), mean_II, -1, Size(r, r));
    mean_II = mean_II / n_mat;

    var_I   = mean_II - mean_I.mul(mean_I);

    double minv, maxv;
    minMaxLoc(p, &minv, &maxv);
    double epsilon = pow((0.001 * (maxv - minv)), 2);

    int r1 = r;
    Mat n1_mat = Mat::ones(I.size(), I.type());
    boxFilter(n1_mat, n1_mat, -1, Size(r1, r1));

    Mat mean_I1, mean_II1, var_I1;
    boxFilter(I, mean_I1, -1, Size(r1, r1));
    mean_I1 = mean_I1 / n1_mat;

    boxFilter(I.mul(I), mean_II1, -1, Size(r1, r1));
    mean_II1 = mean_II1 / n1_mat;
    var_I1   = mean_II1 - mean_I1.mul(mean_I1);

    Mat chi_I;
    sqrt(var_I1.mul(var_I), chi_I);
    Mat weight = (chi_I + epsilon) / (mean(chi_I)[0] + epsilon);

    minMaxLoc(chi_I,&minv,&maxv);
    Mat gamma = (4/(mean(chi_I)[0]-minv)) * (chi_I-mean(chi_I)[0]);
    exp(gamma, gamma);
    gamma = 1.0 - 1.0 / (1.0 + gamma);

    Mat a = (cov_Ip + (eps / weight).mul(gamma)) / (var_I + eps / weight);
    Mat b = mean_p - a.mul(mean_I);

    Mat mean_a, mean_b;
    boxFilter(a, mean_a, -1, Size(r, r));
    mean_a = mean_a / n_mat;

    boxFilter(b, mean_b, -1, Size(r, r));
    mean_b = mean_b / n_mat;

    Mat out = mean_a.mul(I) + mean_b;

	return out;
}
