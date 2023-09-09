#include "pyrfusion.hpp"

MyPyrFusionTest::MyPyrFusionTest() {
}

MyPyrFusionTest::~MyPyrFusionTest() {
}

vector<Mat>  MyPyrFusionTest::GaussianPyramid(Mat img, int level){
    vector<Mat> pyr;
    Mat item = img;
    pyr.push_back(item);
    for (int i = 1; i < level; i++) {
        resize(item, item, item.size()/2, 0, 0, cv::INTER_AREA);
        pyr.push_back(item);
    }
    return pyr;
}

vector<Mat> MyPyrFusionTest::LaplacianPyramid(Mat img, int level) {
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

vector<Mat> MyPyrFusionTest::EdgeFusion(vector<vector<Mat>> pyr_i_arr, vector<vector<Mat>> pyr_w_arr, int n_scales){
	vector<Mat> pyr;
	int img_vec_size = pyr_i_arr.size();
    for(int scale = 0; scale < n_scales; scale++) {
        Mat cur_i = Mat::zeros(pyr_w_arr[0][scale].size(), CV_16SC1);
        for(int i=0; i<cur_i.rows; i++){
            for(int j=0; j<cur_i.cols; j++){
                float value = 0.0, weight = 0.0;
                for (int m = 0; m <img_vec_size; m++) {
                    weight += pyr_w_arr[m][scale].at<uchar>(i,j);
                    value  += pyr_i_arr[m][scale].at<short>(i,j) * pyr_w_arr[m][scale].at<uchar>(i,j);//加残差和融合权重的乘
                }
                value = (value+1) / (weight+1);//加1是为了防止除0
                cur_i.at<short>(i, j) = value;                                                                              
            }
        }
        pyr.push_back(cur_i);
		pyr_i_arr[0][scale].convertTo(pyr_i_arr[0][scale], CV_8UC1);
    }

	return pyr;
}

Mat MyPyrFusionTest::PyrBuild(vector<Mat> pyr, int n_scales) {
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
	out.convertTo(out, CV_8UC1);

	return out;
}

Mat MyPyrFusionTest::Run(vector<Mat> src_arr, vector<Mat> mask_arr) {
	int h = src_arr[0].rows;
    int w = src_arr[0].cols;
    int n_sc_ref = int(log(min(h, w)) / log(2));
    int n_scales = 1;                                                                                                       
    while(n_scales < n_sc_ref) {
        n_scales++;
    }

    int img_vec_size = (int)src_arr.size();
	vector<vector<Mat>> pyr_i_arr, pyr_w_arr;
    for(int i = 0; i< img_vec_size; i++) {
        vector<Mat> pyr_m1 = GaussianPyramid(mask_arr[i], n_scales);
        vector<Mat> pyr_s1 = LaplacianPyramid(src_arr[i], n_scales);
        pyr_w_arr.push_back(pyr_m1);
        pyr_i_arr.push_back(pyr_s1);
    }

	vector<Mat> pyr = EdgeFusion(pyr_i_arr, pyr_w_arr, n_scales);
	Mat out = PyrBuild(pyr, n_scales);

    return out;
}
