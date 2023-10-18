#include "pyr_nlm_blur.hpp"

MyPyrNlmBlurTest::MyPyrNlmBlurTest() {
}

MyPyrNlmBlurTest::~MyPyrNlmBlurTest() {
}

void MyPyrNlmBlurTest::CalLookupTable1(void) {
    for (int i = 0; i < 256; i++) {
        table1[i] = (float)(i*i);
    }
}

void MyPyrNlmBlurTest::CalLookupTable2(void) {
    for (int i = 0; i < 256; i++) {
        for (int j = i; j < 256; j++) {
            table2[i][j] = abs(i - j);
            table2[j][i] = table2[i][j];
        }
    }
}

float MyPyrNlmBlurTest::MseBlock(Mat m1, Mat m2) {
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

Mat MyPyrNlmBlurTest::Nlm(Mat src, float h, int halfKernelSize, int halfSearchSize) {
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
                    int d2 = MseBlock(patchA, patchB);
                    w = exp(-d2/h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                }
            }

            dst_p[i - boardSize] = saturate_cast<uchar>(p / sumw);
        }
    }
    return dst;
}

vector<Mat> MyPyrNlmBlurTest::LaplacianPyramid(Mat img, int level) {
    vector<Mat> pyr;
    Mat item = img;
    for (int i = 1; i < level; i++) {
        Mat item_down;
        Mat item_up;
        resize(item, item_down, item.size()/2, 0, 0, INTER_AREA);
        resize(item_down, item_up, item.size());

        Mat diff(item.size(), CV_16SC1);
        for(int m=0; m<item.rows; m++){
            short *ptr_diff = diff.ptr<short>(m);
            uchar *ptr_up   = item_up.ptr(m);
            uchar *ptr_item = item.ptr(m);

            for(int n=0; n<item.cols; n++){
                ptr_diff[n] = (short)ptr_item[n] - (short)ptr_up[n];//求残差
            }
        }
        pyr.push_back(diff);
        item = item_down;
    }
    item.convertTo(item, CV_16SC1);
    pyr.push_back(item);

    return pyr;
}

Mat MyPyrNlmBlurTest::Run(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    int level = 3;
    vector<Mat> src_arr = LaplacianPyramid(src, level);

    Mat cur_src = src_arr[level-1];
    Mat nlm_blur;
    for(int i=level-1; i>=0; i--) {
        cur_src.convertTo(cur_src, CV_8UC1);
        float cur_h = h / (i+1);
        nlm_blur = Nlm(cur_src, cur_h, halfKernelSize, halfSearchSize);
        if(i>0) {
            resize(nlm_blur, cur_src, src_arr[i-1].size());
            cur_src.convertTo(cur_src, CV_16SC1);
            cur_src = cur_src + src_arr[i-1];
        }
    }

	return nlm_blur;
}
