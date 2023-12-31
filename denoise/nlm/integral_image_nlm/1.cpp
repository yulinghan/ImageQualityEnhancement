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
#include "integral_nlm_blur.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    float h = 8.3;
    int halfKernelSize = 3;
    int halfSearchSize = 9;

    double tickStart = (double)getTickCount();
	MyIntegralNlmBlurTest *my_integra_nlm_blur_test = new MyIntegralNlmBlurTest();
    Mat integra_nlm_blur = my_integra_nlm_blur_test->Run(src, h, halfKernelSize, halfSearchSize);
    double tickEnd = (double)getTickCount();
    cout << "time: " << (tickEnd - tickStart) / (getTickFrequency()) << "s" << endl;

    imshow("integra_nlm_blur", integra_nlm_blur);
    waitKey(0);

    imwrite(argv[2], integra_nlm_blur);

    return 0;
}
