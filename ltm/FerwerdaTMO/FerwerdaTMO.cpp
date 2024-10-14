#include "FerwerdaTMO.hpp"
#include <numeric>

FerwerdaTMO::FerwerdaTMO() {
}

FerwerdaTMO::~FerwerdaTMO() {
}
    
float TpFerwerda(float x) {
    float t = log10(x);

    float y;
    if(t<= -2.6) {
        y = -0.72;
    } else if(t>= 1.9){
        y = t - 1.255;
    } else {
        y = pow((0.249 * t + 0.65), 2.7) - 0.72;
    }

    y = pow(10, y);

    return y;
}

float TsFerwerda(float x) {
    float t = log10(x);

    float y;
    if(t<= -3.94) {
        y = -2.86;
    } else if(t>= -1.44){
        y = t - 0.395;
    } else {
        y = pow((0.405 * t + 1.6), 2.18) - 2.86;
    }

    y = pow(10, y);

    return y;
}

float WalravenValeton_k(float L_wa) {
    float wv_sigma = 100;

    float k = (wv_sigma-L_wa/4) / (wv_sigma+L_wa);
    if(k<0) {
        k = 0.0;
    }

    return k;
}

Mat FerwerdaTMO::Run(Mat src) {
    float Ld_Max = 100;
    float L_da = Ld_Max / 2;

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    double minValue, maxValue;
    minMaxLoc(src_gray, &minValue, &maxValue, NULL, NULL);
    float L_wa = maxValue / 2;
    
    float mc = TpFerwerda(L_da) / TpFerwerda(L_wa);
    float mr = TsFerwerda(L_da) / TsFerwerda(L_wa);
    float k = WalravenValeton_k(L_wa);

    float vec[3] = {1.05, 0.97, 1.27};
    vector<Mat> channels;
    split(src, channels);
    for(int i=0; i<3; i++) {
        channels[i] = mc*channels[i] + vec[i]*mr*k*src_gray;
    }

    Mat out;
    merge(channels, out);

    out = out / Ld_Max;

    return out;
}
