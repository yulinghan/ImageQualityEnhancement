#include "two_scale_photographic.hpp"
#include "poisson_fft.hpp"
#include <numeric>

TwoScalePhotoGraphic::TwoScalePhotoGraphic() {
}

TwoScalePhotoGraphic::~TwoScalePhotoGraphic() {
}


void GetTowScaleImg(Mat src, Mat ref, Mat &base_src, Mat &detail_src, Mat &base_ref, Mat &detail_ref) {

     //双边滤波参数
    int filterDiameter = 5;  // 滤波器直径
    double colorSigma = 30;  // 颜色空间标准差
    double spaceSigma = 70;  // 空间域标准差

    //双边滤波
    bilateralFilter(src, base_src, filterDiameter, colorSigma, spaceSigma);
    detail_src = src - base_src;

    bilateralFilter(ref, base_ref, filterDiameter, colorSigma, spaceSigma);
    detail_ref = ref - base_ref;
}

Mat GradientReversalRemoval(Mat src, Mat detail_src) {
    Mat gxI = Mat::zeros(src.size(), CV_32FC1);
    Mat gyI = Mat::zeros(src.size(), CV_32FC1);
    Mat gxD = Mat::zeros(src.size(), CV_32FC1);
    Mat gyD = Mat::zeros(src.size(), CV_32FC1);
    Mat gx  = Mat::zeros(src.size(), CV_32FC1);
    Mat gy  = Mat::zeros(src.size(), CV_32FC1);

    for(int i=0; i<src.rows-1; i++) {
        for(int j=0; j<src.cols-1; j++) {
            gxI.at<float>(i, j) = src.at<float>(i, j+1) - src.at<float>(i, j);
            gyI.at<float>(i, j) = src.at<float>(i+1, j) - src.at<float>(i, j);
            gxD.at<float>(i, j) = detail_src.at<float>(i, j+1) - detail_src.at<float>(i, j);
            gyD.at<float>(i, j) = detail_src.at<float>(i+1, j) - detail_src.at<float>(i, j);
        }
    }

    for(int i=0; i<src.rows; i++) {
        float *ptr_gxD = gxD.ptr<float>(i);
        float *ptr_gyD = gyD.ptr<float>(i);
        float *ptr_gxI = gxI.ptr<float>(i);
        float *ptr_gyI = gyI.ptr<float>(i);
        float *ptr_gx  = gx.ptr<float>(i);
        float *ptr_gy  = gy.ptr<float>(i);
        for(int j=0; j<src.cols; j++) {
            if((ptr_gxD[j]>0 && ptr_gxI[j]<0) || (ptr_gxD[j]<0 && ptr_gxI[j]>0)) {
                ptr_gx[j] = 0;
            } else if(abs(ptr_gxD[j]) > abs(ptr_gxI[j])) {
                ptr_gx[j] = ptr_gxI[j];
            } else {
                ptr_gx[j] = ptr_gxD[j];
            }

            if((ptr_gyD[j]>0 && ptr_gyI[j]<0) || (ptr_gyD[j]<0 && ptr_gyI[j]>0)) {
                ptr_gy[j] = 0;
            } else if(abs(ptr_gyD[j]) > abs(ptr_gyI[j])) {
                ptr_gy[j] = ptr_gyI[j];
            } else {
                ptr_gy[j] = ptr_gyD[j];
            }
        }
    }
 
    MyPoissonFusionTest *my_poisson_test = new MyPoissonFusionTest();
    Mat out = my_poisson_test->Run(detail_src, gx, gy);

    return out;
}

Mat TwoScalePhotoGraphic::Run(Mat src, Mat ref) {

    resize(src, src, src.size()/4*4);
    resize(ref, ref, ref.size()/4*4);

    src.convertTo(src, CV_32FC1);
    src = src / 255.0;
    ref.convertTo(ref, CV_32FC1);
    ref = ref / 255.0;

    Mat base_src, detail_src, base_ref, detail_ref;
    GetTowScaleImg(src, ref, base_src, detail_src, base_ref, detail_ref);

    imshow("base_src", base_src);
    imshow("base_ref", base_ref);
    imshow("detail_src", abs(detail_src)*2);
    imshow("detail_ref", abs(detail_ref)*2);

    cout << "src:" << src.size() << ", ref:" << ref.size() << endl;

    detail_src = GradientReversalRemoval(src, detail_src);
    base_src = src - detail_src;

    detail_ref = GradientReversalRemoval(ref, detail_ref);
    base_ref = ref - detail_ref;

    imshow("base_src2", base_src);
    imshow("base_ref2", base_ref);
    imshow("detail_src2", abs(detail_src)*2);
    imshow("detail_ref2", abs(detail_ref)*2);

    Mat out;
    return out;
}
