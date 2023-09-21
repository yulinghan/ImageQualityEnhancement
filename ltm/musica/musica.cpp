#include "musica.hpp"

MyMusicaTest::MyMusicaTest() {
}

MyMusicaTest::~MyMusicaTest() {
}

vector<Mat> MyMusicaTest::LaplacianPyramid(Mat img, int level) {
    vector<Mat> pyr;
    Mat item = img;
    for (int i = 1; i < level; i++) {                                                                                                                                         
        Mat item_down;
        Mat item_up;
        resize(item, item_down, item.size()/2, 0, 0, INTER_AREA);
        resize(item_down, item_up, item.size());

        Mat diff(item.size(), CV_8UC1);
        for(int m=0; m<item.rows; m++){
            uchar *ptr_diff = diff.ptr(m);
            uchar *ptr_up   = item_up.ptr(m);
            uchar *ptr_item = item.ptr(m);

            for(int n=0; n<item.cols; n++){
                ptr_diff[n] = ptr_item[n] - ptr_up[n] + 128;
            }
        }
        pyr.push_back(diff);
        item = item_down;
    }
    pyr.push_back(item);

    return pyr;
}                     

Mat MyMusicaTest::PyrBuild(vector<Mat> pyr, int n_scales) {
    Mat out = pyr[n_scales - 1].clone();
    for (int i = n_scales - 2; i >= 0; i--) {
        resize(out, out, pyr[i].size());//上采样
        for(int m = 0; m< out.rows; m++) {
            uchar *ptr_out = out.ptr(m);
            uchar *ptr_i   = pyr[i].ptr(m);
            for(int n = 0; n< out.cols; n++) {
                ptr_out[n] = max(min(ptr_out[n] + (ptr_i[n] - 128), 255), 0);
            }
        }                                                                                                                   
    }
    out.convertTo(out, CV_8UC1);

    return out;
}

vector<Mat> MyMusicaTest::mapping(vector<Mat> pyr_arr, int level, float power) {
    int m = 127;
    vector<double> map;

    for(int i=0; i<256; i++) {
        double detail = i - 128;
        detail = m * (detail / m);

        if(detail >= 0) {
            detail = m * pow(detail/m, power);
        } else {
            detail = -1*m*pow(-1*detail/m, power);
        }
        detail = fmax(fmin(detail + 128.0, 255.0), 0.0);
        map.push_back(detail);
    }

    for(int k=0; k<level; k++) {
        for(int i=0; i<pyr_arr[k].rows; i++) {
            for(int j=0; j<pyr_arr[k].cols; j++) {
                pyr_arr[k].at<uchar>(i, j) = map[pyr_arr[k].at<uchar>(i, j)];
            }
        }
    }
    return pyr_arr;
}

Mat MyMusicaTest::Run(Mat src, float power) {
    int level = 4;
    
    vector<Mat> pyr_arr = LaplacianPyramid(src, level);
    pyr_arr = mapping(pyr_arr, level, power);
    Mat out = PyrBuild(pyr_arr, level);

	return out;
}
