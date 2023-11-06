#include "bior.hpp"

MyBiorTest::MyBiorTest() {
}

MyBiorTest::~MyBiorTest() {
}

void MyBiorTest::bior15_coef( vector<float> &lp1, vector<float> &hp1, vector<float> &lp2, vector<float> &hp2){
    const float coef_norm = 1.f / (sqrtf(2.f) * 128.f);
    const float sqrt2_inv = 1.f / sqrtf(2.f);

    lp1.resize(10);
    lp1[0] =  3.f  ;
    lp1[1] = -3.f  ;
    lp1[2] = -22.f ;
    lp1[3] =  22.f ;
    lp1[4] =  128.f;
    lp1[5] =  128.f;
    lp1[6] =  22.f ;
    lp1[7] = -22.f ;
    lp1[8] = -3.f  ;
    lp1[9] =  3.f  ;

    hp1.resize(10);
    hp1[0] =  0.f;
    hp1[1] =  0.f;
    hp1[2] =  0.f;
    hp1[3] =  0.f;
    hp1[4] = -sqrt2_inv;
    hp1[5] =  sqrt2_inv;
    hp1[6] =  0.f;
    hp1[7] =  0.f;
    hp1[8] =  0.f;
    hp1[9] =  0.f;

    lp2.resize(10);
    lp2[0] = 0.f;
    lp2[1] = 0.f;
    lp2[2] = 0.f;
    lp2[3] = 0.f;
    lp2[4] = sqrt2_inv;
    lp2[5] = sqrt2_inv;
    lp2[6] = 0.f;
    lp2[7] = 0.f;
    lp2[8] = 0.f;
    lp2[9] = 0.f;

    hp2.resize(10);
    hp2[0] =  3.f  ;
    hp2[1] =  3.f  ;
    hp2[2] = -22.f ;
    hp2[3] = -22.f ;
    hp2[4] =  128.f;
    hp2[5] = -128.f;
    hp2[6] =  22.f ;
    hp2[7] =  22.f ;
    hp2[8] = -3.f  ;
    hp2[9] = -3.f  ;

    for (unsigned k = 0; k < 10; k++) {
        lp1[k] *= coef_norm;
        hp2[k] *= coef_norm;
    }
}


Mat MyBiorTest::bior_recover(Mat src) {
    Mat dst = Mat::zeros(src.size(), CV_32FC1);

    return dst;
}

Mat MyBiorTest::bior_decompose(Mat src) {
    Mat out = Mat::zeros(src.size(), CV_32FC1);

    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    return out;
}
