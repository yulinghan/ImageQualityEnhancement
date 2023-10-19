#include "weight_pyr_nlm_blur.hpp"

MyWeightPyrNlmBlurTest::MyWeightPyrNlmBlurTest() {
}

MyWeightPyrNlmBlurTest::~MyWeightPyrNlmBlurTest() {
}

Mat MyWeightPyrNlmBlurTest::GetIntegralImg(Mat src) {
    Mat out;
    src.convertTo(out, CV_32F);

    for (int i=0;i<out.rows;i++) {
        for (int j = 1; j<out.cols; j++) {
            out.at<float>(i, j) += out.at<float>(i, j - 1);
        }
    }

    for (int i = 1; i<out.rows; i++) {
        for (int j = 0; j<out.cols; j++) {
            out.at<float>(i, j) += out.at<float>(i-1, j);
        }
    }
    return out;
}

Mat MyWeightPyrNlmBlurTest::Nlm(Mat src, Mat integral_mat, float h, int hk, int hs) {
    Mat dst = Mat::zeros(src.size(), CV_8UC1);

    for (int i=hs+hk+1; i<src.rows-hs-hk-1; i++) {
        uchar *dst_p = dst.ptr<uchar>(i);
        for (int j=hs+hk+1; j<src.cols-hs-hk-1; j++) {
            float w = 0;
            float p = 0;
            float sumw = 0;

            Mat weight_hs = Mat::zeros(Size(2*hs, 2*hs), CV_32FC1);
            for(int k=0; k<hk; k+=2) {
                float h1 = 1.0 / pow(h, 2);
                float h2 = 1.0 / (2*k+1) / (2*k+1);
                float h3 = h1*h2;

                float center_sum = integral_mat.at<float>(i+k, j+k) + integral_mat.at<float>(i-k-1, j-k-1) 
                                - integral_mat.at<float>(i+k, j-k-1) - integral_mat.at<float>(i-k-1, j+k);

                for(int m=-hs; m<hs; m++) {
                    for(int n=-hs; n<hs; n++) {
                
                        float cur_sum = integral_mat.at<float>(i+m+k, j+n+k) + integral_mat.at<float>(i+m-k-1, j+n-k-1) 
                                            - integral_mat.at<float>(i+m+k, j+n-k-1) - integral_mat.at<float>(i+m-k-1, j+n+k);

                        float sum = (center_sum-cur_sum)*(center_sum-cur_sum);
                        w = exp(-sum*h3);
                        weight_hs.at<float>(m+hs, n+hs) += w;
                    }
                }
            }
            for(int m=-hs; m<hs; m++) {
                for(int n=-hs; n<hs; n++) {
                    p += weight_hs.at<float>(m+hs, n+hs)*src.at<uchar>(i+m, j+n);
                    sumw += weight_hs.at<float>(m+hs, n+hs); 
                }
            }
            dst_p[j] = saturate_cast<uchar>(p / sumw);
        }
    }
    return dst;
}

Mat MyWeightPyrNlmBlurTest::Run(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat integral_mat = GetIntegralImg(src);

    Mat nlm_blur = Nlm(src, integral_mat, h, halfKernelSize, halfSearchSize);

	return nlm_blur;
}
