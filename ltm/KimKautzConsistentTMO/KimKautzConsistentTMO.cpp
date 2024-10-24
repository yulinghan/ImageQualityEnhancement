#include "KimKautzConsistentTMO.hpp"
#include <numeric>

KimKautzConsistentTMO::KimKautzConsistentTMO() {
}

KimKautzConsistentTMO::~KimKautzConsistentTMO() {
}

Mat ChangeLuminance(Mat src, Mat new_l, Mat old_l) {
    Mat out, scale_mat;
    divide(new_l, old_l, scale_mat);

    vector<Mat> channels;
    split(src, channels);
    for(int c=0; c<3; c++) {
        channels[c] = channels[c].mul(scale_mat);
    }

    merge(channels, out);

    return out;
}

Mat KimKautzConsistentTMO::Run(Mat src) {
    float Ld_max = 300;
    float Ld_min = 0.3;
    float KK_c1  = 25.0;
    float KK_c2  = 0.15;

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat l_log;
    log(src_gray + 1e-6, l_log);
    double mu = mean(l_log)[0];

    double max_l, min_l;    // 最大值，最小值
    minMaxLoc(l_log, &min_l, &max_l, NULL, NULL);

    double max_ld = log(Ld_max);
    double min_ld = log(Ld_min);

    float k1 = (max_ld - min_ld) / (max_l - min_l);
    float d0 = max_l - min_l;
    float sigma = d0 / KK_c1;
    float sigma_sq_2 = pow(sigma, 2) * 2;

    Mat l_log2;
    pow(l_log - mu, 2, l_log2);

    Mat w;
    exp(-l_log2 / sigma_sq_2, w);

    Mat k2 = (1 - k1)*w + k1;

    Mat l_d = KK_c2 * k2;
    exp(l_d.mul(l_log - mu) + mu, l_d);

    minMaxLoc(l_d, &min_l, &max_l, NULL, NULL);
    l_d = (l_d - min_l) / (max_l - min_l);

    Mat out = ChangeLuminance(src, l_d, src_gray);

    return out;
}
