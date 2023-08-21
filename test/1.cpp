#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

Mat GetOverMaskUchar(Mat &src_in, float strength, int flag) {
    Mat src;
    if (CV_8UC1 == src_in.type()) {
        src = src_in;
    } else {
        src_in.convertTo(src, CV_8UC1);
    }

    float array_weight1[256], array_weight2[256];
    for (int i = 0; i < 256; i++) {
        float scale = strength * (i - flag);
        array_weight1[i] = exp(scale);
        array_weight2[i] = exp(-scale);
    }

    Mat out = Mat::zeros(src_in.size(), CV_32FC1);
    for (int i = 0; i < src.rows; i++) {
        const uchar *ptr_s = src.ptr<uchar>(i);
        float *ptr_o = out.ptr<float>(i);
        for (int j = 0; j < src.cols; j++) {
            float scale = (array_weight1[ptr_s[j]] - array_weight2[ptr_s[j]]) /
                          (array_weight1[ptr_s[j]] + array_weight2[ptr_s[j]]) +
                          1;
            ptr_o[j] = fmin(fmax(scale * 0.5, 0.0), 1.0);
        }
    }
    return out;
}


std::vector<cv::Mat> GaussianPyramid(cv::Mat img, int level) {
    std::vector<cv::Mat> pyr;
    cv::Mat item = img;

    pyr.push_back(item);
    for (int i = 1; i < level; i++) {
        cv::resize(item, item, item.size()/2, 0, 0, cv::INTER_AREA);
        pyr.push_back(item);
    }
    return pyr;
}

std::vector<cv::Mat> LaplacianPyramid(cv::Mat img, int level) {
    std::vector<cv::Mat> pyr;
    cv::Mat item = img;
    for (int i = 1; i < level; i++) {
        cv::Mat item_down;
        cv::Mat item_up;
        cv::resize(item, item_down, item.size()/2, 0, 0, cv::INTER_AREA);
        cv::resize(item_down, item_up, item.size());

        cv::Mat diff(item.size(), CV_16SC1);
        for(int m=0; m<item.rows; m++){
            for(int n=0; n<item.cols; n++){
                diff.ptr<short>(m)[n] = (short)item.ptr<uchar>(m)[n] - (short)item_up.ptr<uchar>(m)[n];//求残差
            }
        }
        pyr.push_back(diff);
        item = item_down;
    }
    item.convertTo(item, CV_16SC1);
    pyr.push_back(item);
    return pyr;
}


Mat pyrFusion(vector<Mat> src_arr, vector<Mat> mask_arr) {
    int n_sc_ref = int(log(min(src_arr[0].rows, src_arr[0].cols)) / log(2));
    int n_scales = 1;
    while(n_scales < n_sc_ref) {
        n_scales++;
    }

    vector<vector<Mat>> pyr_w_arr, pyr_i_arr;
    for(int i=0; i<(int)src_arr.size(); i++) {
        vector<Mat> pyr_w = GaussianPyramid(mask_arr[i], n_scales);
        vector<Mat> pyr_i = LaplacianPyramid(src_arr[i], n_scales);

        pyr_w_arr.push_back(pyr_w);
        pyr_i_arr.push_back(pyr_i);
    }

    vector<Mat> pyr;
    for (int scale = 0; scale < n_scales; scale++) {
        Mat cur_i = Mat::zeros(pyr_w_arr[0][scale].size(), CV_16SC1);
        for(int i=0; i<cur_i.rows; i++){
            for(int j=0; j<cur_i.cols; j++){
                float value = 0.0, weight = 0.0;
                for (int m = 0; m < (int)pyr_w_arr.size(); m++) {
                    weight += pyr_w_arr[m][scale].at<float>(i,j);
                    value  += pyr_i_arr[m][scale].at<short>(i,j) * pyr_w_arr[m][scale].at<float>(i,j);//加残差和融合权重的乘积
                }
                value = value/ (weight+ 0.01);//加1是为了防止除0
                cur_i.at<short>(i, j) = value;
            }
        }
        pyr.push_back(cur_i);
    }

    Mat res = pyr[n_scales - 1].clone();
    for (int i = n_scales - 2; i >= 0; i--) {
        resize(res, res, pyr[i].size());

        for(int m= 0; m< res.rows; m++) {
            for(int n= 0; n< res.cols; n++) {
                int value = res.at<short>(m, n) + pyr[i].at<short>(m,n);
                res.at<short>(m, n) = value;
            }
        }
    }

    return res;
}

vector<Mat> GetWeight(vector<Mat> src_arr) {
    vector<Mat> weight_arr;

    Mat weight1 = Mat::zeros(src_arr[0].size(), CV_32FC1);
    Mat weight2 = Mat::zeros(src_arr[0].size(), CV_32FC1);
    for(int i=0; i<src_arr[0].rows; i++) {
        for(int j=0; j<src_arr[0].cols; j++) {
            weight1.at<float>(i,j) = (255-src_arr[0].at<uchar>(i,j)) / 255.0;
            weight2.at<float>(i,j) = (src_arr[1].at<uchar>(i,j)) / 255.0;
        }
    }
    
    weight_arr.push_back(weight1);
    weight_arr.push_back(weight2);

    return weight_arr;
}

int main(int argc, char* argv[]) {
    Mat src1 = imread(argv[1], 0);
    Mat src2 = imread(argv[2], 0);
    Mat src3 = imread(argv[3], 0);

    vector<Mat> src_arr;
    src_arr.push_back(src1);
    src_arr.push_back(src2);
    src_arr.push_back(src3);

    imshow("src1", src1);
    imshow("src2", src2);
    imshow("src3", src3);

    Mat weight0 = imread("input/weight0.jpg", 0);
    Mat weight1 = imread("input/weight1.jpg", 0);
    Mat weight2 = imread("input/weight2.jpg", 0);
        
    resize(weight0, weight0, src1.size());
    resize(weight1, weight1, src1.size());
    resize(weight2, weight2, src1.size());

    weight0.convertTo(weight0, CV_32FC1, 1/255.0);
    weight1.convertTo(weight1, CV_32FC1, 1/255.0);
    weight2.convertTo(weight2, CV_32FC1, 1/255.0);
    vector<Mat> weight_arr;
    weight_arr.push_back(weight0);
    weight_arr.push_back(weight1);
    weight_arr.push_back(weight2);

    Scalar mean_val0 = mean(src1);
    Scalar mean_val1 = mean(src2);
    Scalar mean_val2 = mean(src3);
  
    float scale1 = mean_val0[0] / mean_val1[0];
    float scale2 = mean_val1[0] / mean_val2[0];

    scale1 = (scale1-1.0)/2;
    scale2 = (scale2-1.0)/2;

    /* new dark */
    Mat tmp_mat1 = src3 * (1.0+scale2*1);
    Mat tmp_mat2 = src3 * (1.0+scale2*0);

    vector<Mat> dark_arr;
    dark_arr.push_back(tmp_mat1);
    dark_arr.push_back(tmp_mat2);
    vector<Mat> weight_dark_arr;;
    Mat dark_weight = GetOverMaskUchar(tmp_mat1, 0.057, 230);
    weight_dark_arr.push_back(1.0 - dark_weight);
    weight_dark_arr.push_back(dark_weight);

    Mat dark = pyrFusion(dark_arr, weight_dark_arr);
    dark.convertTo(dark, CV_8UC1);
    imshow("dark", dark);
    imshow("dark0", dark_arr[0]);
    imshow("dark1", dark_arr[1]);
    imshow("weight_dark_arr0", weight_dark_arr[0]);
    imshow("weight_dark_arr1", weight_dark_arr[1]);

    /* new middle */
    tmp_mat1 = src3 * (1.0+scale2*2);
    tmp_mat2 = dark;

    vector<Mat> mid_arr;
    mid_arr.push_back(tmp_mat1);
    mid_arr.push_back(tmp_mat2);
    vector<Mat> weight_mid_arr;
    Mat mid_weight = GetOverMaskUchar(tmp_mat1, 0.017, 200);
    weight_mid_arr.push_back(1.0 - mid_weight);
    weight_mid_arr.push_back(mid_weight);

    Mat mid = pyrFusion(mid_arr, weight_mid_arr);
    mid.convertTo(mid, CV_8UC1);
    imshow("mid", mid);

    /* new middle */
    tmp_mat1 = src_arr[1] * (1.0+scale1*2);
    imshow("mid2", tmp_mat1);
    tmp_mat2 = mid;

    vector<Mat> light_arr;
    light_arr.push_back(tmp_mat1);
    light_arr.push_back(tmp_mat2);
    vector<Mat> weight_light_arr;
    Mat light_weight = GetOverMaskUchar(tmp_mat1, 0.017, 200);
    weight_light_arr.push_back(1.0 - light_weight);
    weight_light_arr.push_back(light_weight);

    Mat light = pyrFusion(light_arr, weight_light_arr);
    light.convertTo(light, CV_8UC1);
    imshow("light", light);

    /* new middle */
    tmp_mat1 = src_arr[0];
    tmp_mat2 = light;

    imshow("1111111", tmp_mat1);
    imshow("22222222", src_arr[0]);
    light_arr.clear();
    light_arr.push_back(tmp_mat1);
    light_arr.push_back(tmp_mat2);
    weight_light_arr.clear();
    light_weight = GetOverMaskUchar(tmp_mat1, 0.017, 200);

    Mat erodeStruct = getStructuringElement(MORPH_RECT,Size(15,15));
    dilate(light_weight, light_weight, erodeStruct);

    weight_light_arr.push_back(1.0 - light_weight);
    weight_light_arr.push_back(light_weight);

    light = pyrFusion(light_arr, weight_light_arr);
    light.convertTo(light, CV_8UC1);
    imshow("light2", light);

    Mat out = pyrFusion(src_arr, weight_arr);
    out.convertTo(out, CV_8UC1);
    imshow("out_ori", out);

    src_arr[1] = light;
    src_arr[2] = mid;
    out = pyrFusion(src_arr, weight_arr);
    out.convertTo(out, CV_8UC1);
    imshow("out1", out);
    waitKey(0);

    return 0;
}
