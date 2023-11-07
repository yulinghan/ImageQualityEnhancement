#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>
#include "hadamard.hpp"

int main(int argc, char* argv[]) {
    int nSx_r = 16;
    Mat src = Mat::zeros(Size(1, nSx_r), CV_32FC1);

    cout << "old:" << endl;
    for(int j=0; j<nSx_r; j++){
        src.at<float>(0, j) = j;
        cout << ", " << j;
    }
    cout << endl;

	MyHadamardTest *my_hadamard_test = new MyHadamardTest();
    Mat hadamard =  Mat::zeros(src.size(), src.type());
    my_hadamard_test->hadamard_transform(src, nSx_r, 0, hadamard);
    cout << "new1:" << endl;
    for(int j=0; j<nSx_r; j++){
        cout << ", " << src.at<float>(0, j);
    }
    cout << endl;

    my_hadamard_test->hadamard_transform(src, nSx_r, 0, hadamard);
    float coef = 1.0f / nSx_r;

    cout << "new2:" << endl;
    for(int j=0; j<16; j++){
        cout << ", " << src.at<float>(0, j) * coef;
    }
    cout << endl;

    waitKey(0);

    return 0;
}
