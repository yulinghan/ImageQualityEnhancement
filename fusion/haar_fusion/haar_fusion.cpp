#include "haar_fusion.hpp"

MyHaarFusionTest::MyHaarFusionTest() {
}

MyHaarFusionTest::~MyHaarFusionTest() {
}

// 二维Haar小波分解
void MyHaarFusionTest::harr_wave_decompose(Mat &src, Mat &LL, Mat &LH, Mat &HL, Mat &HH) {
    Mat image;
    src.convertTo(image, CV_32FC1);
    Mat dst = Mat::zeros(image.size(), CV_32FC1);

    int row = image.rows;
    int col = image.cols;
    Mat image1(row, col, CV_32FC1, Scalar::all(0));
    Mat image2(row, col, CV_32FC1,Scalar::all(0));
    int half_row = row / 2;
    int half_col = col / 2;

    for (int i = 0; i < row; ++i) {
        float *psrc = image.ptr<float>(i);
        float *ptmp = image1.ptr<float>(i);
        for (int j = 0; j < half_col; ++j) {
            int a = j << 1;
            ptmp[j] = (psrc[a] + psrc[a + 1]) * 0.5;
            ptmp[j + half_col] = (psrc[a] - psrc[a + 1]) * 0.5;
        }
    }

    for (int i = 0; i < half_row; ++i) {
        float *pcurrent = image1.ptr<float>(2 * i);
        float *pnext = image1.ptr<float>(2 * i + 1);

        float *p1 = image2.ptr<float>(i);
        float *p2 = image2.ptr<float>(i + half_row);

        for (int j = 0; j < col; ++j) {
            p1[j] = (pcurrent[j] + pnext[j])*0.5;
            p2[j] = (pcurrent[j] - pnext[j])*0.5;
        }
    }

    LL = image2(Rect(0, 0, image2.cols / 2, image2.rows / 2));
    LH = image2(Rect(0, image2.rows / 2, image2.cols / 2, image2.rows / 2));
    HL = image2(Rect(image2.cols / 2, 0, image2.cols / 2, image2.rows / 2));
    HH = image2(Rect(image2.cols / 2, image2.rows / 2, image2.cols / 2, image2.rows / 2));
}

Mat MyHaarFusionTest::harr_wave_recover(Mat &LL, Mat &LH, Mat &HL, Mat &HH) {
    Mat dst = Mat::zeros(LL.size(), CV_32FC1);

    int current_rows = LL.rows;
    int current_cols = LL.cols;
    Mat temp_img1(current_rows*2, current_cols*2, CV_32FC1);
    for (int i=0; i<current_rows; i++) {
        float *pup1 = LL.ptr<float>(i);
        float *pup2 = HL.ptr<float>(i);
        float *pdown1 = LH.ptr<float>(i);
        float *pdown2 = HH.ptr<float>(i);
        int temp_i = i * 2;
        float *p1 = temp_img1.ptr<float>(temp_i);
        float *p2 = temp_img1.ptr<float>(temp_i +1);
        for (int j=0; j<current_cols; j++) {
            int temp_j = j+current_cols;
            p1[j] = (pup1[j] + pdown1[j]);
            p2[j] = (pup1[j] - pdown1[j]);
            p1[temp_j] = (pup2[j] + pdown2[j]);
            p2[temp_j] = (pup2[j] - pdown2[j]);
        }
    }
    
    Mat image2(temp_img1.size(), CV_32FC1);
    for (int i=0; i < image2.rows; i++) {
        float *psrc = temp_img1.ptr<float>(i);
        float *ptmp = image2.ptr<float>(i);
        for (int j=0; j<current_cols; j++) {
            int temp_j = j << 1;
            ptmp[temp_j] = psrc[j] + psrc[j + current_cols];
            ptmp[temp_j+1] = psrc[j] - psrc[j + current_cols];
        }
    }

    return image2;
}

Mat MyHaarFusionTest::Run(Mat src) {
    int level = 5;

    src.convertTo(src, CV_32FC1, 1/255.0);

    vector<vector<Mat>>  haar_pyr_arr;
    for(int i=0; i<level; i++) {
        vector<Mat> haar_arr;
        Mat LL, LH, HL, HH;
        harr_wave_decompose(src, LL, LH, HL, HH);

        haar_arr.push_back(LL);
        haar_arr.push_back(LH);
        haar_arr.push_back(HL);
        haar_arr.push_back(HH);

        src = LL.clone();
        haar_pyr_arr.push_back(haar_arr);
    }

    Mat out = haar_pyr_arr[level-1][0];
    for(int i=level-1; i>=0; i--) {
        imshow("LL", out);
        imshow("LH", abs(haar_pyr_arr[i][1])*5);
        imshow("HL", abs(haar_pyr_arr[i][2])*5);
        imshow("HH", abs(haar_pyr_arr[i][3])*5);
        out = harr_wave_recover(out, haar_pyr_arr[i][1], haar_pyr_arr[i][2], haar_pyr_arr[i][3]);
        imshow("out", out);
        waitKey(0);
    }

    return out;
}
