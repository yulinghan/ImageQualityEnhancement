#include "nlm_blur.hpp"

#define ALL_PATCH 1

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

    int mask_size = 15;
    Mat rect_mask_ori1 = Mat::zeros(Size(mask_size, mask_size), CV_8UC1);
    Mat rect_mask_ori2 = Mat::zeros(Size(mask_size, mask_size), CV_8UC1);
    Mat rect_mask_ori3 = Mat::zeros(Size(mask_size, mask_size), CV_8UC1);
    Mat rect_mask_ori4 = Mat::zeros(Size(mask_size, mask_size), CV_8UC1);
    Mat rect_mask_ori5 = Mat::zeros(Size(mask_size, mask_size), CV_8UC1);
    Mat rect_mask_ori6 = Mat::zeros(Size(mask_size, mask_size), CV_8UC1);

    for(int i=0; i<rect_mask_ori1.rows; i++) {
        for(int j=0; j<rect_mask_ori1.cols/2+1; j++) {
            rect_mask_ori1.at<uchar>(i, j) = 255;
        }
    }

    for(int i=0; i<rect_mask_ori2.rows; i++) {
        for(int j=rect_mask_ori2.cols/2; j<rect_mask_ori2.cols; j++) {
            rect_mask_ori2.at<uchar>(i, j) = 255;
        }
    }

    for(int i=0; i<rect_mask_ori3.rows/2+1; i++) {
        for(int j=0; j<rect_mask_ori3.cols; j++) {
            rect_mask_ori3.at<uchar>(i, j) = 255;
        }
    }

    for(int i=rect_mask_ori4.rows/2; i<rect_mask_ori4.rows; i++) {
        for(int j=0; j<rect_mask_ori4.cols; j++) {
            rect_mask_ori4.at<uchar>(i, j) = 255;
        }
    }

    for(int i=0; i<rect_mask_ori5.rows; i++) {
        for(int j=0; j<rect_mask_ori5.cols; j++) {
            if(i==j) {
                rect_mask_ori5.at<uchar>(i, j) = 255;
            }
            if(i-1==j) {
                rect_mask_ori5.at<uchar>(i, j) = 255;
            }
            if(i+1==j) {
                rect_mask_ori5.at<uchar>(i, j) = 255;
            }
        }
    }
    flip(rect_mask_ori5, rect_mask_ori6, 0);

    cout << "rect_mask_ori1:" << endl;
    cout << rect_mask_ori1 << endl;

    cout << "rect_mask_ori2:" << endl;
    cout << rect_mask_ori2 << endl;

    cout << "rect_mask_ori3:" << endl;
    cout << rect_mask_ori3 << endl;

    cout << "rect_mask_ori4:" << endl;
    cout << rect_mask_ori4 << endl;

    cout << "rect_mask_ori5:" << endl;
    cout << rect_mask_ori5 << endl;

    cout << "rect_mask_ori6:" << endl;
    cout << rect_mask_ori6 << endl;

    float h2 = h*h;
    int rows = src.rows;
    int cols = src.cols;

    CalLookupTable1();
    CalLookupTable2();

    for (int j = boardSize; j < boardSize + rows; j++) {
        uchar *dst_p = dst.ptr<uchar>(j - boardSize);

        cout << "!!!! j:" << j << ", boardSize + rows:" << boardSize + rows << endl;

        for (int i = boardSize; i < boardSize + cols; i++) {
            Mat patchA = boardSrc(Range(j - halfKernelSize, j + halfKernelSize + 1), Range(i - halfKernelSize, i + halfKernelSize + 1));
#if ALL_PATCH
            Mat patchA_1;
            patchA.copyTo(patchA_1, rect_mask_ori1);

            Mat patchA_2;
            patchA.copyTo(patchA_2, rect_mask_ori2);

            Mat patchA_3;
            patchA.copyTo(patchA_3, rect_mask_ori3);

            Mat patchA_4;
            patchA.copyTo(patchA_4, rect_mask_ori4);

            Mat patchA_5;
            patchA.copyTo(patchA_5, rect_mask_ori5);

            Mat patchA_6;
            patchA.copyTo(patchA_6, rect_mask_ori6);
#endif
            double w = 0;
            double p = 0;
            double sumw = 0;

            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                uchar *boardSrc_p = boardSrc.ptr<uchar>(j + sr);
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {

                    Mat patchB = boardSrc(Range(j + sr - halfKernelSize, j + sr + halfKernelSize + 1), Range(i + sc - halfKernelSize, i + sc + halfKernelSize + 1));
#if ALL_PATCH
                    Mat patchB_1;
                    patchB.copyTo(patchB_1, rect_mask_ori1);

                    Mat patchB_2;
                    patchB.copyTo(patchB_2, rect_mask_ori2);

                    Mat patchB_3;
                    patchB.copyTo(patchB_3, rect_mask_ori3);

                    Mat patchB_4;
                    patchB.copyTo(patchB_4, rect_mask_ori4);

                    Mat patchB_5;
                    patchB.copyTo(patchB_5, rect_mask_ori5);

                    Mat patchB_6;
                    patchB.copyTo(patchB_6, rect_mask_ori6);
#endif
                    float d2 = MseBlock(patchA, patchB);
                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;

                    if(i==100 && j==100) {
                        cout << "sr:" << sr << ", sc:" << sc << ", w:" << w << endl;
                    }
#if ALL_PATCH
                    d2 = MseBlock(patchA_1, patchB_1) * 1.75;
                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                    if(i==100 && j==100) {
                        cout << "sr:" << sr << ", sc:" << sc << ", w:" << w << endl;
                    }

                    d2 = MseBlock(patchA_2, patchB_2) * 1.75;
                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                    if(i==100 && j==100) {
                        cout << "sr:" << sr << ", sc:" << sc << ", w:" << w << endl;
                    }

                    d2 = MseBlock(patchA_3, patchB_3) * 1.75;
                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                    if(i==100 && j==100) {
                        cout << "sr:" << sr << ", sc:" << sc << ", w:" << w << endl;
                    }

                    d2 = MseBlock(patchA_4, patchB_4) * 1.75;
                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                    if(i==100 && j==100) {
                        cout << "sr:" << sr << ", sc:" << sc << ", w:" << w << endl;
                    }

                    d2 = MseBlock(patchA_5, patchB_5) * 2.5;
                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                    if(i==100 && j==100) {
                        cout << "sr:" << sr << ", sc:" << sc << ", w:" << w << endl;
                    }

                    d2 = MseBlock(patchA_6, patchB_6) * 2.5;
                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                    if(i==100 && j==100) {
                        cout << "sr:" << sr << ", sc:" << sc << ", w:" << w << endl;
                    }
#endif
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
