#include "integral_nlm_blur.hpp"

MyIntegralNlmBlurTest::MyIntegralNlmBlurTest() {
}

MyIntegralNlmBlurTest::~MyIntegralNlmBlurTest() {
}

Mat MyIntegralNlmBlurTest::GetIntegralImg(Mat src) {
    int Height = src.rows;
    int Width = src.cols;

    Mat out;
    src.convertTo(out, CV_32F);
    out = out - 128;

    //横向添加
    for (int i=0;i<Height;i++) {
        for (int j = 1; j < Width; j++) {
            out.at<float>(i, j) += out.at<float>(i, j - 1);
        }
    }
    //竖向相加
    for (int i = 1; i < Height; i++) {
        for (int j = 0; j < Width; j++) {
            out.at<float>(i, j) += out.at<float>(i-1, j);
        }
    }

    return out;
}

Mat MyIntegralNlmBlurTest::Nlm(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    int boardSize = halfKernelSize + halfSearchSize;
    Mat boardSrc;
    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展

    Mat integral_mat = GetIntegralImg(boardSrc);

    float h1 = 1.0 / (h*h);
    float h2 = 1.0 / (2*halfKernelSize+1) / (2*halfKernelSize+1);
    h = h1*h2;

    int rows = src.rows;
    int cols = src.cols;

    for (int j = boardSize; j < boardSize + rows; j++) {
        uchar *dst_p = dst.ptr<uchar>(j - boardSize);
        for (int i = boardSize; i < boardSize + cols; i++) {
            double w = 0;
            double p = 0;
            double sumw = 0;
            float center_patch_sum = integral_mat.at<float>(j-halfKernelSize, i-halfKernelSize) + integral_mat.at<float>(j+halfKernelSize, i+halfKernelSize)
                                - integral_mat.at<float>(j-halfKernelSize, i+halfKernelSize) - integral_mat.at<float>(j+halfKernelSize, i-halfKernelSize);

            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                uchar *boardSrc_p = boardSrc.ptr<uchar>(j + sr);
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {
                    float cur_patch_sum = integral_mat.at<float>(j+sr-halfKernelSize, i+sc-halfKernelSize) + integral_mat.at<float>(j+sr+halfKernelSize, i+sc+halfKernelSize)
                                - integral_mat.at<float>(j+sr-halfKernelSize, i+sc+halfKernelSize) - integral_mat.at<float>(j+sr+halfKernelSize, i+sc-halfKernelSize);

                    float sum = (center_patch_sum - cur_patch_sum) * (center_patch_sum - cur_patch_sum);
                    w = exp(-(sum*h));
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                }
            }

            dst_p[i - boardSize] = saturate_cast<uchar>(p / sumw);
        }
    }
    return dst;
}

Mat MyIntegralNlmBlurTest::Run(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat nlm_blur = Nlm(src, h, halfKernelSize, halfSearchSize);

	return nlm_blur;
}
