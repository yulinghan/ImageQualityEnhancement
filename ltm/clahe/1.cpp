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
#include "Clahe.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);

	MyClaheTest *my_clahe_test = new MyClaheTest();
    Mat dst = my_clahe_test->Run(src, 8, 5.0);
    imwrite(argv[2], dst);

    imshow("src", src);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}
