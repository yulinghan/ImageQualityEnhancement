#include "EntropyBasedContrastEnhancement.hpp"
#include <numeric>

EntropyBasedContrastEnhancementTMO::EntropyBasedContrastEnhancementTMO() {
}

EntropyBasedContrastEnhancementTMO::~EntropyBasedContrastEnhancementTMO() {
}

void EntropyBasedContrastEnhancementTMO::init(Mat src) {
    int height = src.rows;
    int width  = src.cols;

    if(height%2 != 0) {
        height = height+1;
    }
    if(width%2 != 0) {
        width = width+1;
    }

    resize(src, m_img_bgr, Size(width, height), 0, 0, INTER_AREA);
    cvtColor(m_img_bgr, m_img_gray, COLOR_BGR2GRAY);
    cvtColor(m_img_bgr, m_img_hsv , COLOR_BGR2HSV);
}

vector<vector<float>> EntropyBasedContrastEnhancementTMO::spatialHistgram(Mat m_img_gray) {
    int height = m_img_gray.rows;
    int width  = m_img_gray.cols;

    float ratio = (float)height / width;
    float M = pow((m_level*ratio), 0.5);
    float N = pow((m_level*ratio), 0.5);

    vector<Mat> roi_arr;
    int step_height = height / M;
    int step_width  = width / M;

    for(int m=0; m<M; m++) {
        for(int n=0; n<N; n++) {
            int left   = m/M * height;
            int top    = n/N * width;

            left = min(max(left, 0), height-1-step_height);
            top  = min(max(top, 0),  width-1-step_width);
            Mat roi_src = m_img_gray(Rect(top, left, step_width, step_height));
            roi_arr.push_back(roi_src);
        }
    }

    vector<vector<float>> patch_histogram_arr;
    for(int k=0; k<m_level; k++) {
        vector<float> level_histogram;
        for(int i=0; i<roi_arr.size(); i++) {
            int cur_num = 0;
            for(int m=0; m<roi_arr[i].rows; m++) {
                for(int n=0; n<roi_arr[i].cols; n++) {
                    if(roi_arr[i].at<uchar>(m, n) == k) {
                        cur_num += 1;
                    }
                }
            }
            level_histogram.push_back(cur_num);
        }
        patch_histogram_arr.push_back(level_histogram);
    }

    return patch_histogram_arr;
}

void EntropyBasedContrastEnhancementTMO::spatialEntropy(vector<vector<float>> patch_histogram_arr, vector<float> &f_cdf, vector<float> &s_k_arr) {
    for(int i=0; i<patch_histogram_arr.size(); i++) {
        int cur_hist_sum = 0;
        float s_k = 0.0;

        for(int j=0; j<patch_histogram_arr[i].size(); j++) {
            cur_hist_sum += patch_histogram_arr[i][j];
        }

        for(int j=0; j<patch_histogram_arr[i].size(); j++) {
            patch_histogram_arr[i][j] = patch_histogram_arr[i][j] / (cur_hist_sum + m_eps);
            if(patch_histogram_arr[i][j] != 0) {
                s_k += -(patch_histogram_arr[i][j] * log2(patch_histogram_arr[i][j]));
            }
        }
        s_k_arr.push_back(s_k);
    }

    float sum_entropy = 0.0;
    for(int i=0; i<s_k_arr.size(); i++) {
        sum_entropy += s_k_arr[i];
    }

    float sum_f_k = 0.0;
    for(int i=0; i<s_k_arr.size(); i++) {
        s_k_arr[i] = s_k_arr[i] / (sum_entropy - s_k_arr[i] + m_eps);
        sum_f_k += s_k_arr[i];
    }

    for(int i=0; i<s_k_arr.size(); i++) {
        s_k_arr[i] = s_k_arr[i] / (sum_f_k + m_eps);
    }

    float cdf_value = s_k_arr[0];
    f_cdf.push_back(cdf_value);
    for(int i=1; i<s_k_arr.size(); i++) {
        cdf_value = cdf_value + s_k_arr[i];
        f_cdf.push_back(cdf_value);
    }
}

Mat EntropyBasedContrastEnhancementTMO::mapping(vector<float> f_cdf, float yd, float yu, Mat src) {
    vector<float> map_lut;

    for(int i=0; i<f_cdf.size(); i++) {
        float value = f_cdf[i]*(yu-yd) + yd;
        map_lut.push_back(value);
    }

    for(int i=0; i<src.rows; i++) {
        for(int j=0; j<src.cols; j++) {
            src.at<uchar>(i, j*3+0) = map_lut[src.at<uchar>(i, j*3+0)];
            src.at<uchar>(i, j*3+1) = map_lut[src.at<uchar>(i, j*3+1)];
            src.at<uchar>(i, j*3+2) = map_lut[src.at<uchar>(i, j*3+2)];
        }
    }

    return src;
}
    
Mat EntropyBasedContrastEnhancementTMO::domainCoefWeight(Mat src, vector<float> s_k_arr, float gamma) {
    float sum_value = 0;

    for(int i=0; i<s_k_arr.size(); i++) {
        if(s_k_arr[i] != 0) {
            sum_value += - s_k_arr[i] * log2(s_k_arr[i]);
        }
    }
    float alpha = pow(sum_value, gamma);

    Mat weight_out = Mat::zeros(src.size(), CV_32FC1);

    for(int i=0; i<weight_out.rows; i++) {
        for(int j=0; j<weight_out.cols; j++) {
            float value1 = (float)i / weight_out.rows * alpha;
            float value2 = (float)j / weight_out.cols * alpha;
            weight_out.at<float>(i, j) = 1.0 + value1 * value2;
        }
    }

    return weight_out;
}

Mat EntropyBasedContrastEnhancementTMO::DomainEnhance(Mat src, Mat weight_mat) {
    src.convertTo(src, CV_32F, 1/255.0);
    vector<Mat> channels;
    split(src, channels);

    for(int i=0; i<channels.size(); i++) {
        Mat dct_mat;
        dct(channels[i], dct_mat);

        dct_mat = dct_mat.mul(weight_mat);
        idct(dct_mat, channels[i]);
    }

    Mat out;
    merge(channels, out);
    out = out*255;
    out.convertTo(out, CV_8UC1);

    return out;
}

Mat EntropyBasedContrastEnhancementTMO::Run(Mat src) {
    //对比度调整
    init(src);
    vector<vector<float>> patch_histogram_arr = spatialHistgram(m_img_gray);
    vector<float> f_cdf, s_k_arr;
    spatialEntropy(patch_histogram_arr, f_cdf, s_k_arr);
    Mat out1 = mapping(f_cdf, 0.0, 255.0, m_img_bgr);

    //细节增强
    float gamma = 0.5;
    Mat weight_mat = domainCoefWeight(m_img_gray, s_k_arr, gamma);
    Mat out2 = DomainEnhance(out1, weight_mat);

    return out2;
}
