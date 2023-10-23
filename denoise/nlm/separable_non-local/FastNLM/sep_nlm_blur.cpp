#include "sep_nlm_blur.hpp"

MySepNlmBlurTest::MySepNlmBlurTest() {
}

MySepNlmBlurTest::~MySepNlmBlurTest() {
}

float MySepNlmBlurTest::MseBlock(Mat m1, Mat m2) {
    float sum = 0.0;
    float m11 = 0.0;
    float m22 = 0.0;
    float m12 = 0.0;
    for (int j = 0; j < m1.rows; j++) {
        uchar *data1 = m1.ptr<uchar>(j);
        uchar *data2 = m2.ptr<uchar>(j);
        for (int i = 0; i < m1.cols; i++) {
            m11 += data1[i]*data1[i];
            m22 += data2[i]*data2[i];
            m12 += data1[i]*data2[i];
        }
    }
    sum = (m11+m22-2*m12) / (m1.rows*m2.cols);
    return sum;
}

Mat MySepNlmBlurTest::CalF(Mat src, int S, int K, float beta){
    Mat box = Mat::ones(Size(1, 2*K+1), CV_32FC1);
    int M = src.rows;
    int N = src.cols;

    Mat F = Mat::zeros(Size(N, S+1), CV_32FC1);
    for(int mu=0; mu<(S+1); mu++) {
        for(int nu=0; nu<(N-(S+1)+mu); nu++) {
            F.at<float>(mu, nu) = src.at<uchar>(0, nu) * src.at<uchar>(0, mu+nu);
        }
    }

    return F;
}

Mat MySepNlmBlurTest::Nlm(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    int boardSize = halfKernelSize + halfSearchSize;
    Mat boardSrc;
    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展

    float h2 = h*h;

    int rows = src.rows;
    int cols = src.cols;

    for (int j = boardSize; j < boardSize + rows; j++) {
        Mat patchA = boardSrc(Range(j, j+1), Range(0, boardSrc.cols));
        Mat F = CalF(patchA, halfSearchSize, halfKernelSize, h);

        uchar *dst_p = dst.ptr<uchar>(j - boardSize);
        for (int i = boardSize; i < boardSize + cols; i++) {
            double w = 0;
            double p = 0;
            double sumw = 0;
            uchar *boardSrc_p = boardSrc.ptr<uchar>(j);
            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                float d2;
                if(sr>=0)
                    d2 = F.at<float>(0, i) + F.at<float>(0, i+sr) - 2*F.at<float>(abs(sr), i);
                else
                    d2 = F.at<float>(0, i) + F.at<float>(0, i+sr) - 2*F.at<float>(abs(sr), i+sr);

                w = exp(-d2 / h2);
                p += boardSrc_p[i + sr] * w;
                sumw += w;
            }
            dst_p[i - boardSize] = saturate_cast<uchar>(p / sumw);
        }
    }
    return dst;
}

Mat MySepNlmBlurTest::Run(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat nlm_blur_rows = Nlm(src, h, halfKernelSize, halfSearchSize);

    Mat src2 =src.t();
    Mat nlm_blur_cols = Nlm(src2, h, halfKernelSize, halfSearchSize);
    nlm_blur_cols = nlm_blur_cols.t();

    Mat nlm_blur = (nlm_blur_rows + nlm_blur_cols) / 2;

	return nlm_blur;
}
