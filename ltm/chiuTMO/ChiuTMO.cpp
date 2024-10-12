#include "ChiuTMO.hpp"
#include <numeric>

ChiuTMO::ChiuTMO() {
}

ChiuTMO::~ChiuTMO() {
}
    
Mat ChiuTMO::ChiuGlare(Mat Ls, float k, float n, float w) {
    int R = w / 2;

    float kernel[2*R+1][2*R+1];
    float sum_v = 0.0;
    for(int i=0; i<2*R+1; i++) {
        for(int j=0; j<2*R+1; j++) {
            float v = sqrt(pow((float)(i-R)/R, 2) + pow((float)(j-R)/R, 2));
            if(v<=1) {
                float scale = (1.0 - k)*pow(abs(v-1.0), n);
                kernel[i][j] = scale;
                sum_v += scale;
            } else {
                kernel[i][j] = 0;
            }
        }
    }
    Mat kernel_mask = Mat(2*R+1, 2*R+1, CV_32FC1, kernel);
    kernel_mask = kernel_mask / sum_v;
    Point point(-1, -1);

    filter2D(Ls, Ls, -1, kernel_mask, point, 0, BORDER_REFLECT);

    return Ls;
}

Mat ChiuTMO::ChangeLuminance(Mat src, Mat new_l, Mat old_l) {
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

Mat ChiuTMO::ClampAdjust(Mat S, Mat src_gray, float c_clamping) {
    Mat L_inv = 1.0 / src_gray;

    for(int i=0; i<src_gray.rows; i++) {
        for(int j=0; j<src_gray.cols; j++) {
            if(S.at<float>(i, j) >= L_inv.at<float>(i, j)) {
                S.at<float>(i, j) = L_inv.at<float>(i, j);
            }
        }
    }

    float kk[3][3] = {{0.080, 0.113, 0.080}, 
        {0.113, 0.227, 0.113},
        {0.080, 0.113, 0.080}};
    Mat Kore = Mat(3, 3, CV_32FC1, kk);
    Point point(-1, -1);

    for(int i=0; i<c_clamping; i++) {
        filter2D(S, S, -1, Kore, point, 0, BORDER_REFLECT);
    }

    return S;
}

Mat ChiuTMO::Run(Mat src) {
    float k = 0.8, n = 1, w = 3;
    float c_clamping = 500;

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat gauss_mat;
    GaussianBlur(src_gray, gauss_mat, Size(5, 5), 0, 0);
    Mat S = 1.0 / (n * gauss_mat);

    if(c_clamping>0) {
        S = ClampAdjust(S, src_gray, c_clamping);
    }

    Mat Ls = src_gray.mul(S);
    Mat Ld = ChiuGlare(Ls, k, n, w);

    Mat out = ChangeLuminance(src, Ld, src_gray);
    return out;
}
