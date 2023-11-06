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
#include "bior.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

	MyBiorTest *my_bior_test = new MyBiorTest();
    Mat bior = my_bior_test->bior_decompose(src);
    imshow("bior", bior/256);

    Mat bior_recover = my_bior_test->bior_recover(bior);
    imshow("bior_recover", bior_recover/255);

    waitKey(0);

    return 0;
}
