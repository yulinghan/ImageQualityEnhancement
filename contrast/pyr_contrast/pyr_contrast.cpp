#include "pyr_contrast.hpp"

MyPyrContrastTest::MyPyrContrastTest() {
}

MyPyrContrastTest::~MyPyrContrastTest() {
}

vector<Mat> MyPyrContrastTest::LaplacianPyramid(Mat img, int level) {
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

Mat MyPyrContrastTest::PyrBuild(vector<Mat> pyr, int n_scales) {
    Mat out = pyr[n_scales - 1].clone();
    for (int i = n_scales - 2; i >= 0; i--) {
        resize(out, out, pyr[i].size());//上采样
        for(int m = 0; m< out.rows; m++) {
            short *ptr_out = out.ptr<short>(m);
            short *ptr_i   = pyr[i].ptr<short>(m);
            for(int n = 0; n< out.cols; n++) {
                ptr_out[n] = ptr_out[n] + ptr_i[n] * 2.25;
            }
        }                                                                                                                   
    }
    out.convertTo(out, CV_8UC1);

    return out;
}

Mat MyPyrContrastTest::Run(Mat src) {
    int level = 4;
    
    vector<Mat> pyr_arr = LaplacianPyramid(src, level);
    Mat out = PyrBuild(pyr_arr, level);

	return out;
}
