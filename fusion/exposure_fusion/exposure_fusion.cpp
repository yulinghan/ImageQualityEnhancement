#include "exposure_fusion.hpp"

MyExposureFusionTest::MyExposureFusionTest() {
}

MyExposureFusionTest::~MyExposureFusionTest() {
}

Mat MyExposureFusionTest::ContrastCalculate(Mat src) {
    float k[3][3] ={{ 0, -0.125, 0},
                    {-0.125,  0.5, -0.125},
                    { 0, -0.125, 0}};

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    src_gray.convertTo(src_gray, CV_32FC1, 1/255.0);

    Mat contrast_weight;
    Mat Kore = Mat(3, 3, CV_32FC1, k);
    Point point(-1, -1);
    filter2D(src_gray, contrast_weight, -1, Kore, point, 0, BORDER_CONSTANT);
    contrast_weight = abs(contrast_weight);

    return contrast_weight;
}

Mat MyExposureFusionTest::SaturationCalculate(Mat src) {
    Mat src_f;
    src.convertTo(src_f, CV_32F, 1/255.0);

    Mat saturation_weight = Mat::zeros(src.size(), CV_32FC1);
    for(int i=0; i<src.rows; i++) {
        for(int j=0; j<src.cols; j++) {
            float mean_value = (src_f.at<float>(i, j*3+0) + src_f.at<float>(i, j*3+1) + src_f.at<float>(i, j*3+2)) / 3.0;
            float cur_weight = pow(src_f.at<float>(i, j*3+0) - mean_value, 2) + pow(src_f.at<float>(i, j*3+1) - mean_value, 2) + pow(src_f.at<float>(i, j*3+2) - mean_value, 2);
            saturation_weight.at<float>(i, j) = sqrt(cur_weight/3.0);
        }
    }

    return saturation_weight;
}

Mat MyExposureFusionTest::LightCalculate(Mat src) {
    Mat light_weight = Mat::zeros(src.size(), CV_32FC1);
    float lut[256];

    for(int i=0; i<256; i++) {
        float value = pow(i/255.0 - 0.5, 2);
        lut[i] = exp(-value / (2 * (0.2 * 0.2)));
    }

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    for(int i=0; i<src.rows; i++) {
        for(int j=0; j<src.cols; j++) {
            light_weight.at<float>(i, j) = lut[src_gray.at<uchar>(i, j)];
        }
    }

    return light_weight;
}
	
vector<Mat> MyExposureFusionTest::WeightCalculate(vector<Mat> src_arr) {
    vector<Mat> weight_arr;
    float weight_adjust[3] = {1.0, 1.0, 1.0};

    for(int i=0; i<src_arr.size(); i++) {
        Mat contrast_weight = ContrastCalculate(src_arr[i]);
        Mat saturation_weight = SaturationCalculate(src_arr[i]);
        Mat light_weight = LightCalculate(src_arr[i]);

        Mat cur_weight = Mat::zeros(src_arr[i].size(), CV_32FC1);
        for(int i=0; i<cur_weight.rows; i++) {
            for(int j=0; j<cur_weight.cols; j++) {
                cur_weight.at<float>(i, j) = pow(contrast_weight.at<float>(i, j), weight_adjust[0]) 
                                            * pow(saturation_weight.at<float>(i, j), weight_adjust[1])
                                            * pow(light_weight.at<float>(i, j), weight_adjust[2]);
            }
        }
        weight_arr.push_back(cur_weight);
    }
    
    return weight_arr;
}

vector<Mat> MyExposureFusionTest::GaussianPyramid(Mat img, int level){
    vector<Mat> pyr;
    Mat item = img;
    pyr.push_back(item);
    for (int i = 1; i < level; i++) {
        resize(item, item, item.size()/2, 0, 0, cv::INTER_AREA);
        pyr.push_back(item);
    }
    return pyr;
}

vector<Mat> MyExposureFusionTest::LaplacianPyramid(Mat img, int level) {
    vector<Mat> pyr;
    Mat item = img;
    for (int i = 1; i < level; i++) {                                                                                                                                         
        Mat item_down;
        Mat item_up;
        resize(item, item_down, item.size()/2, 0, 0, INTER_AREA);
        resize(item_down, item_up, item.size());

        Mat diff(item.size(), CV_16SC1);
        for(int m=0; m<item.rows; m++){
            short *ptr_diff = diff.ptr<short>(m);
            uchar *ptr_up   = item_up.ptr(m);
            uchar *ptr_item = item.ptr(m);

            for(int n=0; n<item.cols; n++){
                ptr_diff[n] = (short)ptr_item[n] - (short)ptr_up[n];//求残差
            }
        }
        pyr.push_back(diff);
        item = item_down;
    }
    item.convertTo(item, CV_16SC1);
    pyr.push_back(item);

    return pyr;
}

vector<Mat> MyExposureFusionTest::EdgeFusion(vector<vector<Mat>> pyr_i_arr, vector<vector<Mat>> pyr_w_arr, int n_scales){
    vector<Mat> pyr;
    int img_vec_size = pyr_i_arr.size();
    for(int scale = 0; scale < n_scales; scale++) {
        Mat cur_i = Mat::zeros(pyr_w_arr[0][scale].size(), CV_16SC1);
        for(int i=0; i<cur_i.rows; i++){
            for(int j=0; j<cur_i.cols; j++){
                float value = 0.0, weight = 0.0;
                for (int m = 0; m <img_vec_size; m++) {
                    weight += pyr_w_arr[m][scale].at<float>(i,j);
                    value  += pyr_i_arr[m][scale].at<short>(i,j) * pyr_w_arr[m][scale].at<float>(i,j);
                }
                value = value / weight;//加1是为了防止除0
                cur_i.at<short>(i, j) = value;                                                                              
            }
        }
        pyr.push_back(cur_i);
        pyr_i_arr[0][scale].convertTo(pyr_i_arr[0][scale], CV_8UC1);
    }

    return pyr;
}

Mat MyExposureFusionTest::PyrBuild(vector<Mat> pyr, int n_scales) {
    Mat out = pyr[n_scales - 1].clone();
    for (int i = n_scales - 2; i >= 0; i--) {
        resize(out, out, pyr[i].size());//上采样
        for(int m = 0; m< out.rows; m++) {
            short *ptr_out = out.ptr<short>(m);
            short *ptr_i   = pyr[i].ptr<short>(m);
            for(int n = 0; n< out.cols; n++) {
                ptr_out[n] = ptr_out[n] + ptr_i[n];
            }
        }                                                                                                                   
    }
    Mat out2;
    normalize(out, out2, 0, 255, NORM_MINMAX);
    out2.convertTo(out, CV_8UC1);

    return out;
}

Mat MyExposureFusionTest::PyrResult(vector<Mat> src_arr, vector<Mat> weight_arr, int n_scales) {
    vector<vector<Mat>> pyr_i_arr, pyr_w_arr;
    for(int i = 0; i<src_arr.size(); i++) {
        vector<Mat> pyr_m1 = GaussianPyramid(weight_arr[i], n_scales);
        vector<Mat> pyr_s1 = LaplacianPyramid(src_arr[i], n_scales);
        pyr_w_arr.push_back(pyr_m1);
        pyr_i_arr.push_back(pyr_s1);
    }
    vector<Mat> pyr = EdgeFusion(pyr_i_arr, pyr_w_arr, n_scales);
    Mat out = PyrBuild(pyr, n_scales);

    return out;
}

Mat MyExposureFusionTest::PyrFusion(vector<Mat> src_arr, vector<Mat> weight_arr) {
    int h = src_arr[0].rows;
    int w = src_arr[0].cols;
    int n_sc_ref = int(log(min(h, w)) / log(2));
    int n_scales = 1;
    while(n_scales < n_sc_ref) {
        n_scales++;
    }

    vector<Mat> channels_b, channels_g, channels_r;
    for(int i=0; i<weight_arr.size(); i++) {
        vector<Mat> channels;
        split(src_arr[i], channels);
        channels_b.push_back(channels[0]);
        channels_g.push_back(channels[1]);
        channels_r.push_back(channels[2]);
    }

    Mat b = PyrResult(channels_b, weight_arr, n_scales);
    Mat g = PyrResult(channels_g, weight_arr, n_scales);
    Mat r = PyrResult(channels_r, weight_arr, n_scales);

    Mat out;
    vector<Mat> channels = {b, g, r};
    merge(channels, out);

    return out;
}

Mat MyExposureFusionTest::Run(vector<Mat> src_arr) {
	vector<Mat> weight_arr = WeightCalculate(src_arr);
    Mat out = PyrFusion(src_arr, weight_arr);

    return out;
}
