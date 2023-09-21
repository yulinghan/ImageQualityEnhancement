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
#include "alpha.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src1 = imread(argv[1]);
    Mat src2 = imread(argv[2]);
    Mat mask = imread(argv[3], 0);

	MyAlphaTest *my_alpha_test = new MyAlphaTest();
    Mat out = my_alpha_test->Run(src1, src2, mask);

	imwrite(argv[4], out);

    return 0;
}
