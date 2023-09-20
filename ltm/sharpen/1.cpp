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
#include "direction.hpp"
#include "usm.hpp"
#include "multiscalesharpen.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    Mat dst;
	MyUsmTest *my_usm_test = new MyUsmTest();
    dst = my_usm_test->Run(src, 17, 3, 1.0);
    imshow("usm", dst);

	MyDirectionTest *my_direction_test = new MyDirectionTest();
    dst = my_direction_test->Run(src, 17, 1.0);
    imshow("direction", dst);

	MyMultiScaleSharpenTest *my_mult_test = new MyMultiScaleSharpenTest();
    dst = my_mult_test->Run(src, 17, 1.0);
    imshow("multiscalesharpen", dst);

    waitKey(0);

    return 0;
}
