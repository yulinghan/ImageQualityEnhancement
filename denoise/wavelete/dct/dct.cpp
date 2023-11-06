#include "dct.hpp"

MyDctTest::MyDctTest() {
}

MyDctTest::~MyDctTest() {
}

Mat MyDctTest::dct_recover(Mat src) {
    Mat dst = Mat::zeros(src.size(), CV_32FC1);

    double f;
    double Cu, Cv;

    for(int ys = 0; ys < src.rows; ys ++){
        for(int xs = 0; xs < src.cols; xs ++){
            f = 0;

            for (int u = 0; u < src.rows; u++){
                for (int v = 0; v < src.cols; v++){
                    if (u == 0){
                        Cu = sqrt(1.0/src.rows);
                    } else {
                        Cu = sqrt(2.0/src.rows);
                    }

                    if (v == 0){
                        Cv = sqrt(1.0/src.cols);
                    } else {
                        Cv = sqrt(2.0/src.rows);
                    }
                    f += Cu*Cv*src.at<float>(u, v)*cos((2*ys+1)*u*PI/(2.0*src.rows)) 
                        * cos((2*xs+1)*v*PI/(2.0*src.cols));
                }
            }

            f = fmin(fmax(f, 0), 255);
            dst.at<float>(ys, xs) = (uchar)f;
        }
    }

    return dst;
}

Mat MyDctTest::dct_decompose(Mat src) {
    double I;
    double F;
    double Cu, Cv;

    Mat out = Mat::zeros(src.size(), CV_32FC1);
    for (int ys = 0; ys < src.rows; ys ++){
        for (int xs = 0; xs < src.cols; xs ++){
            if (ys == 0){
                Cu = sqrt(1.0/src.rows);
            } else{
                Cu = sqrt(2.0/src.rows);
            }

            if (xs == 0){
                Cv = sqrt(1.0/src.cols);
            }else {
                Cv = sqrt(2.0/src.cols);
            }

            F = 0;
            for (int y = 0; y < src.rows; y++){
                for(int x = 0; x < src.cols; x++){
                    I = (double)src.at<uchar>(y, x);
                    F += I * cos((2*y+1)*ys*PI/(2.0 * src.rows)) * cos((2*x+1)*xs*PI/(2.0 * src.cols));
                }
            }
            out.at<float>(ys, xs) = Cu*Cv*F;
        }
    }

    return out;
}
