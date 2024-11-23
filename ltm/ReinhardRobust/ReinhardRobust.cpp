#include "ReinhardRobust.hpp"
#include <numeric>

ReinhardRobust::ReinhardRobust() {
}

ReinhardRobust::~ReinhardRobust() {
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

float ReinhardAlpha(Mat src_gray, float delta) {
    double minValue, maxValue;
    minMaxLoc(src_gray, &minValue, &maxValue, NULL, NULL);

    float log2Min = log2(minValue + delta);
    float log2Max = log2(maxValue + delta);

    Mat log_src;
    log(src_gray, log_src);

    Scalar mean1 = mean(log_src+delta);
    float logaver  = exp(mean1[0]);
    float log2aver = log2(logaver + delta);

    float cur_v = (2.0*log2aver-log2Min-log2Max)/(log2Max-log2Min);
    float alpha = 0.18 * pow(4, cur_v);

    return alpha;
}

float ReinhardWhite(Mat src_gray, float delta) {
    double minValue, maxValue;
    minMaxLoc(src_gray, &minValue, &maxValue, NULL, NULL);

    float log2min = log2(minValue + delta);
    float log2max = log2(maxValue + delta);
    
    float wp = 1.5 * pow(2, log2max - log2min - 4.5);

    return wp;
}

Mat ReinhardGTM(Mat src_gray) {
    float delta = 1e-6;

    Mat log_src;
    log(src_gray, log_src);

    Scalar mean1 = mean(log_src+delta);
    float Lwa = exp(mean1[0]);

    float p_alpha = ReinhardAlpha(src_gray, delta);
    float p_white = ReinhardWhite(src_gray, delta);

    Mat l_s = p_alpha/Lwa * src_gray;
    
    Mat l_d;
    divide(l_s.mul(1.0+l_s/(p_white*p_white)), (1.0+l_s), l_d);

    return l_d;
}

Mat CalV(Mat src_gray, float pPhi, float alpha) {
    int s_max = 8;

    vector<Mat> gauss_arr;
    gauss_arr.push_back(src_gray.clone());
    for(int i=0; i<s_max; i++) {
        int cur_r = (i+1)*4+1;

        Mat cur_mat;
        GaussianBlur(gauss_arr[i], cur_mat, Size(cur_r, cur_r), 0, 0);
        gauss_arr.push_back(cur_mat);
    }

    float constant = pow(2, pPhi) * alpha;
    float s = pow(1.6, 8);
    float pEps = 0.05;

    vector<Mat> v_arr;
    for(int i=0; i<gauss_arr.size()-1; i++) {
        Mat v1_mat = gauss_arr[i];
        Mat v2_mat = gauss_arr[i+1];
        
        Mat cur_v = (v1_mat - v2_mat);
        float scale = constant / pow(s, 2);
        divide(cur_v, scale+v1_mat, cur_v);
        cur_v = abs(cur_v);
        v_arr.push_back(cur_v);
    }
    
    Mat dst_v = gauss_arr[v_arr.size()-1].clone();

    for(int i=0; i<dst_v.rows; i++) {
        for(int j=0; j<dst_v.cols; j++) {
            for(int k=0; k<v_arr.size(); k++) {
                if(v_arr[k].at<float>(i, j) > pEps) {
                    dst_v.at<float>(i, j) = gauss_arr[k].at<float>(i, j);
                    continue;
                }
            }
        }
    }
    return dst_v;
}

Mat ReinhardLTM(Mat src_gray) {
    float delta = 1e-6;

    Mat log_src;
    log(src_gray, log_src);

    Scalar mean1 = mean(log_src+delta);
    float Lwa = exp(mean1[0]);

    float p_alpha = ReinhardAlpha(src_gray, delta);
    float p_white = ReinhardWhite(src_gray, delta);

    Mat l_s = p_alpha/Lwa * src_gray;

    float pPha = 8.0;
    Mat v_mat = CalV(src_gray, pPha, p_alpha);

    Mat l_d;
    divide(l_s.mul(1.0+l_s/(p_white*p_white)), (1.0+v_mat), l_d);
   
    return l_d;
}

Mat ReinhardRobust::Run(Mat src) {
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    double minValue, maxValue;
    minMaxLoc(src_gray, &minValue, &maxValue, NULL, NULL);

    float delta = 1e-6;
    float log2Min = log2(minValue + delta);
    float log2Max = log2(maxValue + delta);
    cout << "log2Max:" << log2Max << ", log2Min:" << log2Min << endl;

    Mat out;
    if(log2Max - log2Min > 11) {
        out = ReinhardLTM(src_gray);
    } else {
        out = ReinhardGTM(src_gray);
    }

    out = ChangeLuminance(src, out, src_gray);

    return out;
}
