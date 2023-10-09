#include "recursive_bilateral.hpp"

MyRecursiveBilateral::MyRecursiveBilateral() {
}

MyRecursiveBilateral::~MyRecursiveBilateral() {
}

Mat MyRecursiveBilateral::HorizontalFiltering(Mat src, double *range_table, double alpha) {
    Mat out1 = src.clone();
    Mat out2 = src.clone();

    double  ycr, ypr;
    for(int i=0; i<src.rows; i++) {
        uchar *ptr_src  = src.ptr<uchar>(i);
        uchar *ptr_out1 = out1.ptr<uchar>(i);
        uchar *ptr_out2 = out2.ptr<uchar>(i);

        ycr = ypr = ptr_src[0];
        for(int j=1; j<src.cols; j++) {
            int dr = abs(ptr_src[j] - ptr_src[j-1]);
            double weight= range_table[dr];
            double alpha_= weight*alpha;
            ptr_out1[j] = ycr = (1.0 - alpha_) * ptr_src[j] + alpha_* ypr;
            ypr = ycr;
        }

        ycr = ypr = ptr_out1[src.cols-1];
        for(int j=src.cols-2; j>=0; j--) {
            int dr = abs(ptr_out1[j] - ptr_out1[j+1]);
            double weight= range_table[dr];
            double alpha_= weight*alpha;
            ycr = (1.0 - alpha_) * ptr_out1[j] + alpha_* ypr;
            ptr_out2[j] = (ycr + ptr_out1[j]) /2;
//            ptr_out2[j] = ycr;
            ypr = ycr;
        }
    }
    
    imshow("rrr", out2);

    Mat out3 = src.clone();
    Mat out4 = src.clone();
    for(int i=0; i<src.cols; i++) {
        ycr = ypr = out2.at<uchar>(0, i);
        for(int j=1; j<src.rows; j++) {
            int dr = abs(out2.at<uchar>(j, i) - out2.at<uchar>(j-1, i));
            double weight= range_table[dr];
            double alpha_= weight*alpha;
            out3.at<uchar>(j, i) = ycr = (1.0 - alpha_) * out2.at<uchar>(j, i) + alpha_* ypr;
            ypr = ycr;
        }

        ycr = ypr = out3.at<uchar>(0, i);
        for(int j=src.rows-2; j>=0; j--) {
            int dr = abs(out3.at<uchar>(j, i) - out3.at<uchar>(j+1, i));
            double weight= range_table[dr];
            double alpha_= weight*alpha;
            ycr = (1.0 - alpha_) * out3.at<uchar>(j, i) + alpha_* ypr;
            out4.at<uchar>(j, i) = (ycr + out3.at<uchar>(j, i)) / 2;
            ypr = ycr;
        }
    }

    return out4;
}


Mat MyRecursiveBilateral::Run(Mat src, float sigma_spatial, float sigma_range) {
    int h = src.rows;
    int w = src.cols;
    int max_pixel = 255;

    double alpha = exp(-sqrt(2.0)/(sigma_spatial*w));
    double range_table[max_pixel+1];
    double inv_sigma_range = 1.0/(sigma_range*max_pixel);
    for(int i=0;i<=max_pixel; i++) {
        range_table[i]=exp(-i*inv_sigma_range);
    }

    Mat out = HorizontalFiltering(src, range_table, alpha);

	return out;
}
