#include "recursive_bilateral.hpp"

MyRecursiveBilateral::MyRecursiveBilateral() {
}

MyRecursiveBilateral::~MyRecursiveBilateral() {
}

Mat MyRecursiveBilateral::BilateralFiltering(Mat src, double *range_table, float sigma_spatial) {
    Mat out1 = Mat::zeros(src.size(), CV_64FC1);
    Mat out2 = Mat::zeros(src.size(), CV_64FC1);
    Mat factor_mat1 = Mat::zeros(src.size(), CV_64FC1);
    Mat factor_mat2 = Mat::zeros(src.size(), CV_64FC1);
    
    double alpha = exp(-sqrt(2.0)/(sigma_spatial*src.cols));

    double ypr;
    for(int i=0; i<src.rows; i++) {
        uchar  *ptr_src    = src.ptr<uchar>(i);
        double *ptr_out1   = out1.ptr<double>(i);
        double *ptr_out2   = out2.ptr<double>(i);
        double *ptr_factor1= factor_mat1.ptr<double>(i);
        double *ptr_factor2= factor_mat2.ptr<double>(i);

        ypr = ptr_src[0];
        double fp = 1.0;
        ptr_out1[0] = ptr_src[0];
        ptr_factor1[0] = fp;
        for(int j=1; j<src.cols; j++) {
            int dr = abs(ptr_src[j] - ptr_src[j-1]);
            double weight  = range_table[dr];
            double alpha_  = weight*alpha;
            ptr_out1[j]    = ypr = (1.0 - alpha) * ptr_src[j] + alpha_* ypr;
            ptr_factor1[j] = fp = (1.0 - alpha) + alpha_*fp;

        }
        ypr = ptr_out1[src.cols-1] / ptr_factor1[src.cols-1];

        fp = 1.0;
        ptr_out2[src.cols-1] = ptr_src[src.cols-1];
        ptr_factor2[src.cols-1] = fp;
        for(int j=src.cols-2; j>=0; j--) {
            int dr = abs(ptr_src[j] - ptr_src[j+1]);
            double weight  = range_table[dr];
            double alpha_  = weight*alpha;

            ptr_out2[j]    = ypr = (1.0 - alpha) * ptr_src[j] + alpha_* ypr;
            ptr_factor2[j] = fp = (1.0 - alpha) + alpha_*fp;
        }
    }
    
    out2 = (out1 + out2) / 2;
    factor_mat2 = (factor_mat1 + factor_mat2) / 2;
    Mat tmp = out2 / factor_mat2;
    tmp.convertTo(tmp, CV_8UC1);
    imshow("tmp", tmp);

    alpha = exp(-sqrt(2.0) / (sigma_spatial*src.rows));//filter kernel size

    Mat out3 = out2.clone();
    Mat out4 = out2.clone();;
    Mat factor_mat3 = factor_mat2.clone();
    Mat factor_mat4 = factor_mat2.clone();
    
    for(int i=1; i<src.rows; i++) {
        for(int j=0; j<src.cols; j++) {
            int dr = abs(src.at<uchar>(i, j) - src.at<uchar>(i-1, j));
            double weight  = range_table[dr];
            double alpha_  = weight*alpha;
            out3.at<double>(i, j) = (1.0 - alpha)*out2.at<double>(i, j) + alpha_* out3.at<double>(i-1, j);
            factor_mat3.at<double>(i, j) = (1.0 - alpha)*factor_mat2.at<double>(i, j) + alpha_* factor_mat3.at<double>(i-1, j);
        }
    }

    for(int i=src.rows-2; i>=0; i--) {
        for(int j=0; j<src.cols; j++) {
            int dr = abs(src.at<uchar>(i, j) - src.at<uchar>(i+1, j));
            double weight  = range_table[dr];
            double alpha_  = weight*alpha;
            out4.at<double>(i, j) = (1.0 - alpha)*out2.at<double>(i, j) + alpha_* out4.at<double>(i+1, j);
            factor_mat4.at<double>(i, j) = (1.0 - alpha)*factor_mat2.at<double>(i, j) + alpha_* factor_mat4.at<double>(i+1, j);
        }
    }
    
    out4 = (out3 + out4) / 2;
    factor_mat4 = (factor_mat3 + factor_mat4) / 2;
    out4 = out4 / factor_mat4;

    Mat out = out4;
    out.convertTo(out, CV_8UC1);

    return out;
}


Mat MyRecursiveBilateral::Run(Mat src, float sigma_spatial, float sigma_range) {
    int max_pixel = 255;
    double range_table[max_pixel+1];
    double inv_sigma_range = 1.0/(sigma_range*max_pixel);
    for(int i=0;i<=max_pixel; i++) {
        range_table[i]=exp(-i*inv_sigma_range);
    }

    Mat out = BilateralFiltering(src, range_table, sigma_spatial);

	return out;
}
