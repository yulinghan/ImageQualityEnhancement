#include "MyTonemapReinhard.hpp"

MyTonemapReinhard::MyTonemapReinhard() {
}

MyTonemapReinhard::~MyTonemapReinhard() {
}

Mat MyTonemapReinhard::linear(Mat src, float gamma)  {
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

void MyTonemapReinhard::log_(const Mat& src, Mat& dst){
    max(src, Scalar::all(1e-4), dst);
    log(dst, dst);
}

Mat MyTonemapReinhard::Run(Mat src, float gamma, float intensity, 
                            float light_adapt, float color_adapt) {
    imshow("src", src);
    src = linear(src, 1.0);

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat log_img;
    log_(src_gray, log_img);

    float log_mean = sum(log_img)[0] / log_img.total();
    double log_min, log_max;
    minMaxLoc(log_img, &log_min, &log_max);

    double key = (log_max - log_mean) / (log_max - log_min);
    float map_key = 0.3f + 0.7f * pow(key, 1.4f);
    intensity = exp(-intensity);
    Scalar chan_mean = mean(src);
    float gray_mean = mean(src_gray)[0];

    vector<Mat> channels(3);
    split(src, channels);

    for(int i = 0; i < 3; i++) {
        float global = color_adapt*chan_mean[i] + (1.0f-color_adapt) * gray_mean;
        Mat adapt = color_adapt * channels[i] + (1.0f - color_adapt) * src_gray;
        adapt = light_adapt * adapt + (1.0f - light_adapt) * global;
        pow(intensity * adapt, map_key, adapt);
        channels[i] = channels[i].mul(1.0f / (adapt + channels[i]));
    }

    Mat dst;
    merge(channels, dst);

    dst = linear(dst, gamma);

    imshow("dst", dst);
    waitKey(0);

    return dst;
}
