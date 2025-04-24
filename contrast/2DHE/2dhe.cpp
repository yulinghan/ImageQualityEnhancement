#include "2dhe.hpp"

My2DHETest::My2DHETest() {
}

My2DHETest::~My2DHETest() {
}

Mat Calcu_2DHE(Mat src, int window_size) {
    int histSize = 256;

    vector<vector<float>> hist_vec;
    hist_vec.resize(histSize);
    for (auto& row : hist_vec) {
        row.resize(histSize, 0);
    }

    for(int i=0; i<src.rows; i++) {
        for(int j=0; j<src.cols; j++) {
            int ori_value = src.at<uchar>(i, j);

            for(int m=i-window_size; m<i+window_size; m++) {
                int addr_i = m;
                if(addr_i<0) {
                    addr_i = 0;
                }
                if(addr_i>=src.rows) {
                    addr_i = src.rows - 1;
                }
                for(int n=j-window_size; n<j+window_size; n++) {
                    int addr_j = n;
                    if(addr_j<0) {
                        addr_j = 0;
                    }
                    if(addr_j>=src.cols) {
                        addr_j = src.cols - 1;
                    }
                    int cur_value = src.at<uchar>(addr_i, addr_j);
                    hist_vec[ori_value][cur_value] += abs(ori_value-cur_value)+1;
                }
            }
        }
    }

    float all_num = 0;
    for(int i=0; i<histSize; i++) {
        for(int j=0; j<histSize; j++) {
            all_num += hist_vec[i][j];
        }
    }

    for(int i=0; i<histSize; i++) {
        for(int j=0; j<histSize; j++) {
            hist_vec[i][j] = hist_vec[i][j] / all_num;
        }
    }

    vector<float> cur_hist;
    cur_hist.resize(histSize);

    for(int i=0; i<histSize; i++) {
        float cur_hist_value = 0;
        for(int j=0; j<histSize; j++) {
            cur_hist_value += hist_vec[i][j];
        }
        cur_hist[i] = cur_hist_value;
        if(i>0) {
            cur_hist[i] += cur_hist[i-1];
        }
    }

    Mat dst = Mat::zeros(src.size(), CV_8UC1);

    for(int i=0; i<src.rows; i++) {
        for(int j=0; j<src.cols; j++) {
            dst.at<uchar>(i, j) = cur_hist[src.at<uchar>(i, j)] * 255;
        }
    }

    return dst;
}

Mat My2DHETest::Run(Mat src) {
    int window_size = 31;

    Mat cur_src;
    src.convertTo(cur_src, CV_8UC1);
    Mat out = Calcu_2DHE(cur_src, window_size);
    imshow("cur_src", cur_src);
    imshow("out_gray", out);
    out.convertTo(out, CV_32FC1);

    Mat result;
    equalizeHist(cur_src, result);
    imshow("result", result);

    Mat weight;
    divide(out+1, src+1, weight);

	return weight;
}
