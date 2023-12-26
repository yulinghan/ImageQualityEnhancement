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
#include "pull_push_nlm.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    float h = 2.1;
    int halfKernelSize = 1;
    int halfSearchSize = 3;
	MyPullPushNlmTest *my_pull_nlm_test = new MyPullPushNlmTest();
    my_pull_nlm_test->PullNlm(src, h, halfKernelSize, halfSearchSize);

    waitKey(0);

    return 0;
}
