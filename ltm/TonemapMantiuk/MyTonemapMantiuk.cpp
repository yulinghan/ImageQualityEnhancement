#include "MyTonemapMantiuk.hpp"

MyTonemapMantiuk::MyTonemapMantiuk() {
}

MyTonemapMantiuk::~MyTonemapMantiuk() {
}

Mat MyTonemapMantiuk::linear(Mat src, float gamma)  {
    Mat dst;

    double min, max;
    minMaxLoc(src, &min, &max);
    if(max - min > DBL_EPSILON) {
        dst = (src - min) / (max - min);
    } else {
        src.copyTo(dst);
    }

    pow(dst, 1.0f / gamma, dst);

    return dst;
}

void MyTonemapMantiuk::log_(const Mat& src, Mat& dst){
    max(src, Scalar::all(1e-4), dst);
    log(dst, dst);
}

void MyTonemapMantiuk::getGradient(Mat src, Mat& dst, int pos) {
    dst = Mat::zeros(src.size(), CV_32F);
    Mat a, b;
    Mat grad = src.colRange(1, src.cols) - src.colRange(0, src.cols - 1);
    grad.copyTo(dst.colRange(pos, src.cols + pos - 1));
    if(pos == 1) {
        src.col(0).copyTo(dst.col(0));
    }
}

void MyTonemapMantiuk::getContrast(Mat src, std::vector<Mat>& x_contrast, std::vector<Mat>& y_contrast)
{
    int levels = static_cast<int>(logf(static_cast<float>(min(src.rows, src.cols))) / logf(2.0f));
    x_contrast.resize(levels);
    y_contrast.resize(levels);

    Mat layer;
    src.copyTo(layer);
    for(int i = 0; i < levels; i++) {
        getGradient(layer, x_contrast[i], 0);
        getGradient(layer.t(), y_contrast[i], 0);
        resize(layer, layer, Size(layer.cols / 2, layer.rows / 2), 0, 0, INTER_LINEAR);
    }
}

void MyTonemapMantiuk::signedPow(Mat src, float power, Mat& dst) {
    Mat sign = (src > 0);
    sign.convertTo(sign, CV_32F, 1.0f/255.0f);
    sign = sign * 2.0f - 1.0f;
    pow(abs(src), power, dst);
    dst = dst.mul(sign);
}

void MyTonemapMantiuk::mapContrast(Mat& contrast) {
    const float response_power = 0.4185f;
    signedPow(contrast, response_power, contrast);
    contrast *= scale;
    signedPow(contrast, 1.0f / response_power, contrast);
}

void MyTonemapMantiuk::calculateSum(std::vector<Mat>& x_contrast, std::vector<Mat>& y_contrast, Mat& sum) {
    if (x_contrast.empty())
        return;
    const int last = (int)x_contrast.size() - 1;
    sum = Mat::zeros(x_contrast[last].size(), CV_32F);

    for(int i = last; i >= 0; i--) {
        Mat grad_x, grad_y;
        getGradient(x_contrast[i], grad_x, 1);
        getGradient(y_contrast[i], grad_y, 1);
        resize(sum, sum, x_contrast[i].size(), 0, 0, INTER_LINEAR);
        sum += grad_x + grad_y.t();
    }
}

void MyTonemapMantiuk::calculateProduct(Mat src, Mat& dst) {
    std::vector<Mat> x_contrast, y_contrast;
    getContrast(src, x_contrast, y_contrast);
    calculateSum(x_contrast, y_contrast, dst);
}

Mat MyTonemapMantiuk::mapLuminance(Mat src, Mat lum, Mat new_lum, float saturation) {
    Mat dst;
    std::vector<Mat> channels(3);
    split(src, channels);
    for(int i = 0; i < 3; i++) {
        channels[i] = channels[i].mul(1.0f / lum);
        pow(channels[i], saturation, channels[i]);
        channels[i] = channels[i].mul(new_lum);
    }
    merge(channels, dst);

    return dst;
}

Mat MyTonemapMantiuk::Run(Mat src, float gamma, float power, float saturation) {
    Mat out;
    scale = power;

    imshow("src", src);
    src = linear(src, gamma);

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat log_img;
    log_(src_gray, log_img);

    std::vector<Mat> x_contrast, y_contrast;
    getContrast(log_img, x_contrast, y_contrast);

    for(size_t i = 0; i < x_contrast.size(); i++) {
        mapContrast(x_contrast[i]);
        mapContrast(y_contrast[i]);
    }

    Mat right(src.size(), CV_32F);
    calculateSum(x_contrast, y_contrast, right);

    Mat p, r, product, x = log_img;
    calculateProduct(x, r);
    imshow("right", abs(right));
    imshow("r", abs(r));
    r = right - r;
    r.copyTo(p);

    imshow("p", abs(p));
    waitKey(0);

    const float target_error = 1e-3f;
    float target_norm = static_cast<float>(right.dot(right)) * powf(target_error, 2.0f);
    int max_iterations = 100;
    float rr = static_cast<float>(r.dot(r));

    for(int i = 0; i < max_iterations; i++) {
        calculateProduct(p, product);
        double dprod = p.dot(product);
        float alpha = rr / dprod;

        r -= alpha * product;
        x += alpha * p;

        float new_rr = r.dot(r);
        CV_Assert(fabs(rr) > 0);
        p = r + (new_rr / rr) * p;
        rr = new_rr;

        if(rr < target_norm) {
            break;
        }
    }

    exp(x, x);
    Mat dst = mapLuminance(src, src_gray, x, saturation);

    imshow("dst", dst);
    waitKey(0);

    return dst;
}
