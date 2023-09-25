#include "sef.hpp"

MySefTest::MySefTest() {
}

MySefTest::~MySefTest() {
}

float MySefTest::MedianCal(Mat src) {
    float mid_num = src.rows * src.cols / 2;

    int histSize = 256;
    float range[] = { 0,255 };
    const float* histRanges = { range };
    Mat hist_arr;

    calcHist(&src, 1, 0, Mat(), hist_arr, 1, &histSize, &histRanges, true, false);

    float cur_num = 0;
    for(int i=0; i<histSize; i++) {
        cur_num += hist_arr.at<float>(i, 0);
        if(cur_num > mid_num) {
            mid_num = i/255.0;
            break;
        }
    }

    return mid_num;
}

void MySefTest::FrameParamCal(Mat src, float a, float b, float mid_value, int &Mp, int &Ns){
    Mp    = 1;
    Ns    = floor(Mp * mid_value);
    int N = Mp - Ns;
    int Nx= max(N, Ns);

    cout << "!!! mid_value:" << mid_value << endl;

    float tmax1  = (1.0 + (Ns + 1.0) * (b - 1.0) / Mp) / (pow(a, 1.0 / Nx));
    float tmin1s = (-b  + (Ns - 1.0) * (b - 1.0) / Mp) / (pow(a, 1.0 / Nx)) + 1.0;   
    float tmax0  = 1.0 + Ns * (b - 1.0) / Mp;
    float tmin0  = 1.0 - b + Ns * (b - 1.0) / Mp;
    while (tmax1 < tmin0 || tmax0 < tmin1s) {
        Mp++;
        Ns = floor(Mp * mid_value);
        N  = Mp - Ns;
        Nx = max(N, Ns);
        tmax1  = (1.0 + (Ns + 1.0) * (b - 1.0) / Mp) / (pow(a, 1.0 / Nx));
        tmin1s = (-b  + (Ns - 1.0) * (b - 1.0) / Mp) / (pow(a, 1.0 / Nx)) + 1.0;
        tmax0  = 1.0 + Ns * (b - 1.0) / Mp;
        tmin0  = 1.0 - b + Ns * (b - 1.0) / Mp;

        if (Mp > 49) {
            cout << "The estimation of the number of image error!" << endl;
        }
    }
}

vector<Mat> MySefTest::FusionMatCal(Mat src, float a, float b, int Mp, int Ns) {
    int N  = Mp - Ns;
    int Nx = max(N, Ns);
    float lambda = 0.125;
    float b1 = b / 2 + lambda;
    float b2 = b / 2 - lambda;

    vector<Mat> out_arr;
    for(int i=-Ns; i<N; i++) {
        float r = (1.0 - b/2) - (i+Ns)*(1-b)/Mp;

        Mat f;
        if(i<0){
            f = 1.0 + (src-1.0) * pow(a, -i/(float)Nx);
        } else {
            f = src * pow(a, i/(float)Nx);
        }

        Mat fr_diff = f - r;
        Mat abs_fr_diff = abs(fr_diff);

        Mat mask1     = (abs_fr_diff <= b/2);
        mask1.convertTo(mask1, CV_32F, 1.0 / 255.0);

        Mat mask2     = (abs_fr_diff > b/2);
        mask2.convertTo(mask2, CV_32F, 1.0 / 255.0);

        Mat sign = fr_diff.mul(abs_fr_diff);
        sign.setTo(1.0,  sign>0.0);
        sign.setTo(-1.0, sign<0.0);

        Mat b3 = (abs_fr_diff == b);
        b3.convertTo(b3, CV_32F, 1.0 / 255.0);

        Mat g1 = b1 - (lambda * lambda) / (abs_fr_diff - b2 + b3);
        Mat cur_mat = mask1.mul(f) + mask2.mul(sign.mul(g1) + r);
        normalize(cur_mat, cur_mat, 0.0, 1.0, cv::NORM_MINMAX);

        cur_mat = cur_mat *255;
        cur_mat.convertTo(cur_mat, CV_8UC1);
        out_arr.push_back(cur_mat);
    }

    return out_arr;
}


vector<Mat> MySefTest::Run(Mat src, float a, float b) {
    Mat out;
    
    //计算输入图像中值信息
    float mid_value = MedianCal(src);
    src.convertTo(src, CV_32FC1, 1/255.0);

    int Mp, Ns;
    FrameParamCal(src, a, b, mid_value, Mp, Ns);
    vector<Mat> sef_arr = FusionMatCal(src, a, b, Mp, Ns);

	return sef_arr;
}
