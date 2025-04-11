#include "bpbe.hpp"

MyBPBETest::MyBPBETest() {
}

MyBPBETest::~MyBPBETest() {
}

Mat EqualizeHistManual(Mat src) {
    // 获取图像的尺寸和灰度级范围
    int rows = src.rows;
    int cols = src.cols;
    int histSize = 256; // 灰度级数量

    // 初始化直方图和累积直方图
    float hist[histSize] = {0};
    float cdf1[histSize] = {0};
    float cdf2[histSize] = {0};

    Scalar meanValue = mean(src);
    int mean_value = meanValue[0];

    cout << "mean_value:" << mean_value << endl;

    // 计算直方图
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int intensity = src.at<uchar>(i, j);
            hist[intensity]++;
        }
    }

    int num1 = 0, num2 = 0;
    for (int i = 0; i < histSize; ++i) {
        if(i<mean_value) {
            num1 += hist[i];
        } else {
            num2 += hist[i];
        }
    }

    // 计算累积直方图
    cdf1[0] = hist[0];
    for (int i = 1; i < mean_value; ++i) {
        cdf1[i] = cdf1[i - 1] + hist[i];
    }

    cdf2[0] = hist[mean_value];
    for (int i = mean_value+1; i < histSize; ++i) {
        cdf2[i-mean_value] = cdf2[i-1-mean_value] + hist[i];
    }

    for (int i = 0; i < mean_value; ++i) {
        cdf1[i] = cdf1[i] / num1;
    }

    for (int i = mean_value; i < histSize; ++i) {
        cdf2[i-mean_value] = cdf2[i-mean_value] / num2;
    }

    // 映射灰度级
    Mat dst = cv::Mat::zeros(src.size(), src.type());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int intensity = src.at<uchar>(i, j);
            if(intensity<mean_value) {
                dst.at<uchar>(i, j) = cdf1[intensity] * mean_value;
            } else {
                dst.at<uchar>(i, j) = mean_value + cdf2[intensity-mean_value] * (histSize - 1 - mean_value);
            }
        }
    }

    return dst;
}

Mat MyBPBETest::Run(Mat src) {
    Mat cur_src;
    src.convertTo(cur_src, CV_8UC1);
    Mat out = EqualizeHistManual(cur_src);
    imshow("cur_src", cur_src);
    imshow("out_gray", out);
    out.convertTo(out, CV_32FC1);

    Mat result;
    equalizeHist(cur_src, result);
    imshow("result", result);

    Mat weight;
    divide(out+1, src+1, weight);

	return weight;
}
