#include "AshikhminTMO.hpp"

AshikhminTMO::AshikhminTMO(float gamma, int ashikhmin_smax) {
    m_gamma = gamma;
    m_ashikhmin_smax = ashikhmin_smax;
}

AshikhminTMO::~AshikhminTMO() {
}

Mat AshikhminTMO::linear(Mat src, float gamma)  {
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

void AshikhminTMO::log_(const Mat& src, Mat& dst){
    max(src, Scalar::all(1e-4), dst);
    log(dst, dst);
}

void AshikhminTMO::AshikhminFiltering(Mat src_gray, Mat &L, Mat &Ldetail) {
    int Ashikhmin_sMax = m_ashikhmin_smax;

    int size[3] = {src_gray.rows, src_gray.cols, Ashikhmin_sMax};
    
    Mat Lfiltered(src_gray.size().width, src_gray.size().height, CV_32FC(Ashikhmin_sMax));
    Mat LC(src_gray.size().width, src_gray.size().height, CV_32FC(Ashikhmin_sMax));

    vector<Mat> L_filtered_arr, LC_arr;
    split(Lfiltered, L_filtered_arr);
    split(LC, LC_arr);
    for(int i=0; i<Ashikhmin_sMax; i++) {
        int blur_r = i;
        if(blur_r<3) {
            blur_r = 3;
        }
        if(blur_r%2==0) {
            blur_r += 1;
        }
        GaussianBlur(src_gray, L_filtered_arr[i], Size(blur_r, blur_r), 0, 0);

        Mat cur_gauss_mat;
        GaussianBlur(src_gray, cur_gauss_mat, Size(blur_r*2+1, blur_r*2+1), 0, 0);
        divide(cur_gauss_mat, L_filtered_arr[i], cur_gauss_mat);

        LC_arr[i] = abs(L_filtered_arr[i] - cur_gauss_mat);
    }

    float threshold = 0.5;

    Mat L_adapt = 0 - Mat::ones(src_gray.size(), CV_32FC1);
    for(int i=0; i<Ashikhmin_sMax; i++) {
        Mat lc_i = LC_arr[i];
        Mat ind = (lc_i<threshold);
        Mat Lfiltered_i = L_filtered_arr[i];
        Lfiltered_i.copyTo(L_adapt, ind);
    }
    
    Mat ind = (L_adapt<0);
    L_filtered_arr[Ashikhmin_sMax-1].copyTo(L_adapt, ind);
    
    L = L_adapt;
    divide(src_gray, L_adapt, Ldetail);
}

float TVI_Ashikhmin(double L) {
    if(L<0.0034) {
        L = L / 0.0014;
    } else if(L>=0.0034 && L < 1.0) {
        L = 2.4483 + log(L/0.0034) / 0.4027;
    } else if(L>=1.0 && L<7.2444) {
        L = 16.5630 + (L-1) / 0.4027;
    } else {
        L = 32.0693 + log(L/7.2444)/0.0556;
    }

    return L;
}

Mat GatTV_L(Mat L) {
    Mat out = Mat::zeros(L.size(), CV_32FC1);

    for(int i=0; i<out.rows; i++) {
        for(int j=0; j<out.cols; j++) {
            out.at<float>(i, j) = TVI_Ashikhmin(L.at<float>(i, j));
        }
    }

    return out;
}

Mat ChangeLuminance(Mat src, Mat Ld, Mat src_gray) {
    Mat factor = Ld / src_gray;

    vector<Mat> channels;
    split(src, channels);

    for(int i=0; i<channels.size(); i++) {
        channels[i] = channels[i].mul(factor);
    }

    Mat out;
    merge(channels, out);

    return out;
}

Mat AshikhminTMO::Run(Mat src) {
    float LdMax = 255;
    src = linear(src, m_gamma);
  
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat L, Ldetail;
    AshikhminFiltering(src_gray, L, Ldetail);
    imshow("base", L);
    imshow("Ldetail", Ldetail);

    //这里偷懒了，应该用直方图排序，选99.95%像素最为极大值，选0.05%像素作为极小值，避免异常极值干扰
    double minValue, maxValue;    // 最大值，最小值
    minMaxLoc(L, &minValue, &maxValue, NULL, NULL);

    float maxL_TVI = TVI_Ashikhmin(maxValue);
    float minL_TVI = TVI_Ashikhmin(minValue);

    Mat tv_l = GatTV_L(L);

    Mat Ld = (LdMax/255)*(tv_l - minL_TVI) / (maxL_TVI-minL_TVI);
    Ld = Ld.mul(Ldetail);
    imshow("Ld1", Ld);

    Mat out = ChangeLuminance(src, Ld, src_gray);

    return out;
}
