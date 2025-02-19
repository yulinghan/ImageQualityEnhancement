#include "wls.hpp"
#include <numeric>

MyWLS::MyWLS() {
}

MyWLS::~MyWLS() {
}

void CalcuGradient(Mat src, Mat &dx_forward, Mat &dx_backward, Mat &dy_forward, Mat &dy_backward) {
    dx_forward  = Mat::zeros(src.size(), CV_32FC1);
    dx_backward = Mat::zeros(src.size(), CV_32FC1);
    dy_forward  = Mat::zeros(src.size(), CV_32FC1);
    dy_backward = Mat::zeros(src.size(), CV_32FC1);

    for(int i=1; i<src.rows-1; i++) {
        for(int j=1; j<src.cols-1; j++) {
            dx_forward.at<float>(i, j)  = src.at<float>(i, j) - src.at<float>(i, j+1);
            dx_backward.at<float>(i, j) = src.at<float>(i, j) - src.at<float>(i, j-1);
            dy_forward.at<float>(i, j)  = src.at<float>(i, j) - src.at<float>(i+1, j);
            dy_backward.at<float>(i, j) = src.at<float>(i, j) - src.at<float>(i-1, j);
        }
    }
}

Mat process_difference_operator(Mat src, float lambda, float alpha, float epsilon) {
    Mat difference_out;
    pow(abs(src), alpha, difference_out);

    difference_out = -lambda / (epsilon + difference_out);

    return difference_out;
}

Mat MyWLS::Run(Mat src) {
    float ambda=0.35, alpha=1.2, epsilon=1e-4;
 
    Mat src_32f;
    src.convertTo(src_32f, CV_32FC1);

    Mat src_log;
    log(src_32f, src_log);

    Mat dx_forward, dx_backward, dy_forward, dy_backward;
    CalcuGradient(src_log, dx_forward, dx_backward, dy_forward, dy_backward);

    Mat dx_forward_weighted  = process_difference_operator(dx_forward, ambda, alpha, epsilon);
    for(int i=0; i<dx_forward_weighted.cols; i++) {
        dx_forward_weighted.at<float>(dx_forward_weighted.rows-1, i) = 0;
    }

    Mat dx_backward_weighted = process_difference_operator(dx_backward, ambda, alpha, epsilon);
    for(int i=0; i<dx_backward_weighted.cols; i++) {
        dx_backward_weighted.at<float>(0, i) = 0;
    }

    Mat dy_forward_weighted  = process_difference_operator(dy_forward, ambda, alpha, epsilon);
    for(int i=0; i<dy_forward_weighted.rows; i++) {
        dy_forward_weighted.at<float>(i, dy_forward_weighted.cols-1) = 0;
    }

    Mat dy_backward_weighted = process_difference_operator(dy_backward, ambda, alpha, epsilon);
    for(int i=0; i<dy_backward_weighted.rows; i++) {
        dy_backward_weighted.at<float>(i, 0) = 0;
    }

    Mat central_element = 1.0 - (dx_forward_weighted + dx_backward_weighted + dy_forward_weighted + dy_backward_weighted);

    //Form sparse matrix
    int N    = src.rows*src.cols;
    Mat A    = Mat::zeros(Size(1, N*5), CV_32FC1);
    Mat B    = A.clone();
    Mat data = A.clone();

    cout << "0000" << endl;
    float* central_element_data      = reinterpret_cast<float*>(central_element.data);
    float* dx_forward_weighted_data  = reinterpret_cast<float*>(dx_forward_weighted.data);
    float* dx_backward_weighted_data = reinterpret_cast<float*>(dx_backward_weighted.data);
    float* dy_forward_weighted_data  = reinterpret_cast<float*>(dy_forward_weighted.data);
    float* dy_backward_weighted_data = reinterpret_cast<float*>(dy_backward_weighted.data);
    for(int i=0; i<N; i++) {
        A.at<float>(i+N*0, 0)    = i;
        B.at<float>(i+N*0, 0)    = i;
        data.at<float>(i, 0) = central_element_data[i];

        A.at<float>(i+N*1, 0)    = i;
        B.at<float>(i+N*1, 0)    = i + 1;
        data.at<float>(i+N*1, 0) = dx_forward_weighted_data[i];

        A.at<float>(i+N*2, 0)    = i;
        B.at<float>(i+N*2, 0)    = i - 1;
        data.at<float>(i+N*2, 0) = dx_backward_weighted_data[i];

        A.at<float>(i+N*3, 0)    = i;
        B.at<float>(i+N*3, 0)    = i + src.cols;
        data.at<float>(i+N*3, 0) = dx_backward_weighted_data[i];

        A.at<float>(i+N*4, 0)    = i;
        B.at<float>(i+N*4, 0)    = i - src.cols;
        data.at<float>(i+N*4, 0) = dx_backward_weighted_data[i];
    }
    cout << "0001" << endl;

    A.setTo(0, B>=N);
    A.setTo(0, B<0);
    data.setTo(0, B>=N);
    data.setTo(0, B<0);
    B.setTo(0, B>=N);
    B.setTo(0, B<0);

    Mat mat_a = Mat::zeros(Size(N, N), CV_32FC1);
    for(int i=0; i<B.rows; i++) {
        int addr_x = A.at<float>(i, 0);
        int addr_y = B.at<float>(i, 0);
        mat_a.at<float>(addr_x, addr_y) = data.at<float>(i, 0);
    }

    Mat mat_b = Mat::zeros(Size(1, N), CV_32FC1);
    float* src_data      = reinterpret_cast<float*>(src.data);
    for(int i=0; i<N; i++) {
        mat_b.at<float>(i, 0) = src_data[i];
    }

    Mat out = dy_backward_weighted;
    return out;
}
