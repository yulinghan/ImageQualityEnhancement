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
#include "steerable_fusion.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    MySteerableFusionTest *my_steerable_fusion_test = new MySteerableFusionTest();
        
	Mat out = my_steerable_fusion_test->Run(src);

//    imshow("out", out);
    waitKey(0);

    return 0;
}
