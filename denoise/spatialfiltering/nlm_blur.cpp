#include "nlm_blur.hpp"

MyNlmBlurTest::MyNlmBlurTest() {
}

MyNlmBlurTest::~MyNlmBlurTest() {
}

void MyNlmBlurTest::CalLookupTable1(void) {
    for (int i = 0; i < 256; i++) {
        table1[i] = (float)(i*i);
    }
}

void MyNlmBlurTest::CalLookupTable2(void) {
    for (int i = 0; i < 256; i++) {
        for (int j = i; j < 256; j++) {
            table2[i][j] = abs(i - j);
            table2[j][i] = table2[i][j];
        }
    }
}

float MyNlmBlurTest::MseBlock(Mat m1, Mat m2) {
    float sum = 0.0;
    for (int j = 0; j < m1.rows; j++) {
        uchar *data1 = m1.ptr<uchar>(j);
        uchar *data2 = m2.ptr<uchar>(j);
        for (int i = 0; i < m1.cols; i++) {
            sum += table1[table2[data1[i]][data2[i]]];
        }
    }
    sum = sum / (m1.rows*m2.cols);
    return sum;
}

Mat MyNlmBlurTest::Nlm(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    int boardSize = halfKernelSize + halfSearchSize;
    Mat boardSrc;
    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展

    float h2 = h*h;

    int rows = src.rows;
    int cols = src.cols;

    CalLookupTable1();
    CalLookupTable2();

    for (int j = boardSize; j < boardSize + rows; j++) {
        uchar *dst_p = dst.ptr<uchar>(j - boardSize);
        for (int i = boardSize; i < boardSize + cols; i++) {
            Mat patchA = boardSrc(Range(j - halfKernelSize, j + halfKernelSize), Range(i - halfKernelSize, i + halfKernelSize));
            double w = 0;
            double p = 0;
            double sumw = 0;

            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                uchar *boardSrc_p = boardSrc.ptr<uchar>(j + sr);
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {
                    Mat patchB = boardSrc(Range(j + sr - halfKernelSize, j + sr + halfKernelSize), Range(i + sc - halfKernelSize, i + sc + halfKernelSize));
                    float d2 = MseBlock(patchA, patchB);

                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                }
            }

            dst_p[i - boardSize] = saturate_cast<uchar>(p / sumw);
        }
    }
    return dst;
}

Mat MyNlmBlurTest::Run(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat nlm_blur = Nlm(src, h, halfKernelSize, halfSearchSize);

	return nlm_blur;
}
