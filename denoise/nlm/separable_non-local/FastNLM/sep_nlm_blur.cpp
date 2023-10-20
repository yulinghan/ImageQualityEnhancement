#include "sep_nlm_blur.hpp"

MySepNlmBlurTest::MySepNlmBlurTest() {
}

MySepNlmBlurTest::~MySepNlmBlurTest() {
}

float MySepNlmBlurTest::MseBlock(Mat m1, Mat m2) {
    float sum = 0.0;
    for (int j = 0; j < m1.rows; j++) {
        uchar *data1 = m1.ptr<uchar>(j);
        uchar *data2 = m2.ptr<uchar>(j);
        for (int i = 0; i < m1.cols; i++) {
            sum += (data1[i]-data2[i])*(data1[i]-data2[i]);
        }
    }
    sum = sum / (m1.rows*m2.cols);
    return sum;
}

Mat MySepNlmBlurTest::CalF(Mat src, int S, int K, float beta){
    Mat box = Mat::ones(Size(1, 2*K+1));
    int M = src.rows;
    int N = src.clos;
    int L = 4*S+1;

    float L_NN = S*N + N + 0.5*S*(2*K-S-1);
    float L_tot = L_NN + (N-S) + S*K + N*(S-1) - 0.5*S*(S-1);

    Mat F = Mat::zeros(Size(L_tot, L));
    Mat F_bar = Mat::zeros(Size(L_tot, L));
    
    int l = 0;

    for(int i=0; i<(S+1); i++) {
        for(int j=0; j<(N-(S+1)+i); j++) {
            l = l+1;
            for(int lp=0; lp<(2*S+1); lp++) {
            
            }
        }
    }
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

Mat MySepNlmBlurTest::Run(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat nlm_blur = Nlm(src, h, halfKernelSize, halfSearchSize);

	return nlm_blur;
}
