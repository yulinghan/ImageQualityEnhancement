#include "brisk.hpp"
#include "brisk_feature_detector.hpp"
#include "brisk_feature_descriptors.hpp"

MyBriskTest::MyBriskTest() {
}

MyBriskTest::~MyBriskTest() {
}

Mat MyBriskTest::CornersShow(Mat src, vector<KeyPoint> corners) {
    Mat dst = src.clone();

    for(int i=0; i<corners.size(); i++) {
        circle(dst, corners[i].pt, 3, Scalar(255, 0, 0), -1);  // 画半径为1的圆(画点）
    }

    return dst;
}

void MyBriskTest::run(Mat src) {
    int threshold = 60;
    int octaves = 4;
    MyBriskFeatureDetectorTest *my_brisk_feature_detect_test = new MyBriskFeatureDetectorTest();
    vector<KeyPoint> corners = my_brisk_feature_detect_test->run(src, threshold, octaves);
    Mat corner_show = CornersShow(src, corners);
    imshow("corner_show", corner_show);

    float _patternScale = 1.0f;
    MyBriskFeatureDescriptorsTest *my_brisk_feature_descriptors_test = new MyBriskFeatureDescriptorsTest(_patternScale);
    Mat descriptors = my_brisk_feature_descriptors_test->run(src, corners);
    cout << "descriptors:" << descriptors.size() << endl;
}
