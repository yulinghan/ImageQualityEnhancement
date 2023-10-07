#include "bilateral_pyr_blur.hpp"

MyBilateralPyrBlurTest::MyBilateralPyrBlurTest() {
}

MyBilateralPyrBlurTest::~MyBilateralPyrBlurTest() {
}

Mat MyBilateralPyrBlurTest::CalGaussianTemplate(int r, float sigma) {
    float pi = 3.1415926;
    int center = r;
    int ksize = r*2+1;
    float x2, y2;

    Mat Kore = Mat::zeros(Size(ksize, ksize), CV_32FC1);
    for (int i = 0; i < ksize; i++) {
        x2 = pow(i - center, 2);
        for (int j = 0; j < ksize; j++) {
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * pi * sigma;
            Kore.at<float>(i, j) = g;
        }
    }

    return Kore;
}

vector<float> MyBilateralPyrBlurTest::CalValueTemplate(float sigma) {
    vector<float> val_weight_arr;

    for(int i=0; i<256; i++) {
        float cur_weight = exp(-(i*i) / (2 * sigma * sigma));
        val_weight_arr.push_back(cur_weight);
    }

    return val_weight_arr;
}
    
Mat MyBilateralPyrBlurTest::BilateralBlur(Mat src, Mat gaussian_kore, vector<float> val_weight_arr, int r) {
    Mat out = Mat::zeros(src.size(), src.type());

    for (int i=r; i<src.rows-r; i++) {
        for (int j=r; j<src.cols-r; j++) {
            float cur_weight = 0.0;
            float value = 0.0;
            for (int m=-r; m<=r; m++){
				for (int n=-r; n<=r; n++){
                    float weight = gaussian_kore.at<float>(m+r, n+r) 
                            * val_weight_arr[abs(src.at<uchar>(i,j) - src.at<uchar>(i+m,j+n))];
                    value += src.at<uchar>(i+m,j+n) * weight;
                    cur_weight += weight;
                }
            }
            value = value / cur_weight;
            out.at<uchar>(i, j) = value;
        }
    }
    return out;
}

vector<Mat> MyBilateralPyrBlurTest::LaplacianPyramid(Mat img, int level) {
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

Mat MyBilateralPyrBlurTest::Run(Mat src, int r, float gauss_sigma, float value_sigma) {
    Mat gaussian_kore = CalGaussianTemplate(r, gauss_sigma);
    vector<float> val_weight_arr = CalValueTemplate(value_sigma);

    int level = 3;
    vector<Mat> src_arr = LaplacianPyramid(src, level);

    Mat cur_src = src_arr[level-1];
    Mat bilateral_blur;
    for(int i=level-1; i>=0; i--) {
        cur_src.convertTo(cur_src, CV_8UC1);
        bilateral_blur = BilateralBlur(cur_src, gaussian_kore, val_weight_arr, r);
        if(i>0) {
            resize(bilateral_blur, cur_src, src_arr[i-1].size());
            cur_src.convertTo(cur_src, CV_16SC1);
            cur_src = cur_src + src_arr[i-1];
        }
    }

	return bilateral_blur;
}
