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
#include "dct.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    resize(src, src, src.size()/4);
    imshow("src", src);

	MyDctTest *my_dct_test = new MyDctTest();
    Mat dct = my_dct_test->dct_decompose(src);
    imshow("dct", dct/256);

    Mat dct_recover = my_dct_test->dct_recover(dct);
    imshow("dct_recover", dct_recover/255);

    waitKey(0);

    return 0;
}
