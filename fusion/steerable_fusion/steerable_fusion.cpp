#include "steerable_fusion.hpp"

MySteerableFusionTest::MySteerableFusionTest() {
}

MySteerableFusionTest::~MySteerableFusionTest() {
}

void MySteerableFusionTest::Shift(Mat &src) {
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    Mat q0(src, Rect(0, 0, cx, cy));   // ROI区域的左上
    Mat q1(src, Rect(cx, 0, cx, cy));  // ROI区域的右上
    Mat q2(src, Rect(0, cy, cx, cy));  // ROI区域的左下
    Mat q3(src, Rect(cx, cy, cx, cy)); // ROI区域的右下
 
    //交换象限（左上与右下进行交换）
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    //交换象限（右上与左下进行交换）
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);   
}

Mat MySteerableFusionTest::myfft(Mat &src) {
    Mat src2 = Mat::zeros(src.size(), CV_32F);
    Mat planes[] = {src, src2};

    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);

    return complexI;
}

void MySteerableFusionTest::GetFFTFreq(Mat complexI, int r, Mat &fft_low, vector<Mat> &fft_freq_arr) {
    Mat mask_l     = Mat::zeros(complexI.size(), CV_32FC1);
    Mat mask_freq1 = Mat::zeros(complexI.size(), CV_32FC1);
    Mat mask_freq2 = Mat::zeros(complexI.size(), CV_32FC1);
    Mat mask_freq3 = Mat::zeros(complexI.size(), CV_32FC1);
    Mat mask_freq4 = Mat::zeros(complexI.size(), CV_32FC1);

    Mat element = getStructuringElement(MORPH_RECT, Size(17, 17));

    Point point(complexI.cols/2, complexI.rows/2);
    circle(mask_l, point, r, Scalar(1.0), -1);
    GaussianBlur(mask_l, mask_l, Size(57, 57), 0, 0);

    for(int i=0; i<mask_freq1.rows; i++) {
        for(int j=0; j<mask_freq1.cols; j++) {
            if(i == mask_freq1.cols-j) {
                mask_freq1.at<float>(i, j) = 1.0;
            }
        }
    }
    dilate(mask_freq1, mask_freq1, element);
    mask_freq1 = mask_freq1 - mask_l;
    mask_freq1.setTo(0.0, mask_freq1<0);
    GaussianBlur(mask_freq1, mask_freq1, Size(57, 57), 0, 0);
    imshow("mask_freq1", mask_freq1);

    for(int i=0; i<mask_freq2.rows; i++) {
        for(int j=0; j<mask_freq2.cols; j++) {
            if(i == j) {
                mask_freq2.at<float>(i, j) = 1.0;
            }
        }
    }
    dilate(mask_freq2, mask_freq2, element);
    mask_freq2 = mask_freq2 - mask_l;
    mask_freq2.setTo(0.0, mask_freq2<0);
    GaussianBlur(mask_freq2, mask_freq2, Size(57, 57), 0, 0);
    imshow("mask_freq2", mask_freq2);

    for(int i=0; i<mask_freq3.rows; i++) {
        for(int j=0; j<mask_freq3.cols; j++) {
            if(i == mask_freq3.rows/2) {
                mask_freq3.at<float>(i, j) = 1.0;
            }
        }
    }
    dilate(mask_freq3, mask_freq3, element);
    mask_freq3 = mask_freq3 - mask_l;
    mask_freq3.setTo(0.0, mask_freq3<0);
    GaussianBlur(mask_freq3, mask_freq3, Size(57, 57), 0, 0);
    imshow("mask_freq3", mask_freq3);

    for(int i=0; i<mask_freq4.rows; i++) {
        for(int j=0; j<mask_freq4.cols; j++) {
            if(j == mask_freq4.cols/2) {
                mask_freq4.at<float>(i, j) = 1.0;
            }
        }
    }
    dilate(mask_freq4, mask_freq4, element);
    mask_freq4 = mask_freq4 - mask_l;
    mask_freq4.setTo(0.0, mask_freq4<0);
    GaussianBlur(mask_freq4, mask_freq4, Size(57, 57), 0, 0);
    imshow("mask_freq4", mask_freq4);

    vector<Mat> planes;
    split(complexI.clone(), planes);
    Shift(planes[0]);
    Shift(planes[1]);
    planes[0] = planes[0].mul(mask_l);
    planes[1] = planes[1].mul(mask_l);
    Shift(planes[0]);
    Shift(planes[1]);
    merge(planes, fft_low);

    Mat dst_freq1, dst_freq2, dst_freq3, dst_freq4;
    split(complexI.clone(), planes);
    Shift(planes[0]);
    Shift(planes[1]);
    planes[0] = planes[0].mul(mask_freq1);
    planes[1] = planes[1].mul(mask_freq1);
    Shift(planes[0]);
    Shift(planes[1]);
    merge(planes, dst_freq1);
    fft_freq_arr.push_back(dst_freq1);

    split(complexI.clone(), planes);
    Shift(planes[0]);
    Shift(planes[1]);
    planes[0] = planes[0].mul(mask_freq2);
    planes[1] = planes[1].mul(mask_freq2);
    Shift(planes[0]);
    Shift(planes[1]);
    merge(planes, dst_freq2);
    fft_freq_arr.push_back(dst_freq2);

    split(complexI.clone(), planes);
    Shift(planes[0]);
    Shift(planes[1]);
    planes[0] = planes[0].mul(mask_freq3);
    planes[1] = planes[1].mul(mask_freq3);
    Shift(planes[0]);
    Shift(planes[1]);
    merge(planes, dst_freq3);
    fft_freq_arr.push_back(dst_freq3);

    split(complexI.clone(), planes);
    Shift(planes[0]);
    Shift(planes[1]);
    planes[0] = planes[0].mul(mask_freq4);
    planes[1] = planes[1].mul(mask_freq4);
    Shift(planes[0]);
    Shift(planes[1]);
    merge(planes, dst_freq4);
    fft_freq_arr.push_back(dst_freq4);
}

Mat MySteerableFusionTest::myifft(Mat src) {
    Mat re;
    idft(src, re);
    re = re / re.rows / re.cols;
    Mat re_channels[2];
    split(re, re_channels);

    Mat result;
    re_channels[0] = abs(re_channels[0]);
    re_channels[0].convertTo(result, CV_8UC1);

    return result;
}

Mat MySteerableFusionTest::Run(Mat src) {
    src.convertTo(src, CV_32FC1);

    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);

    Mat padded;
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_WRAP);

    cout << "padded:" << padded.size() << endl;

    Mat complexI = myfft(padded);

    Mat fft_low;
    vector<Mat> fft_freq_arr;

    int r = min(complexI.rows, complexI.cols) / 8;
    GetFFTFreq(complexI, r, fft_low, fft_freq_arr);

	Mat out = myifft(fft_low);
    imshow("low", out);

	out = myifft(fft_freq_arr[0]);
    imshow("freq0", out*5);

	out = myifft(fft_freq_arr[1]);
    imshow("freq1", out*5);

	out = myifft(fft_freq_arr[2]);
    imshow("freq2", out*5);

	out = myifft(fft_freq_arr[3]);
    imshow("freq3", out*5);

    return out;
}
