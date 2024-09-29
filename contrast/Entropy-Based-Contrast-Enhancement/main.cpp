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
#include "EntropyBasedContrastEnhancement.hpp"

int main(int argc, char* argv[]) {
    Mat img = imread(argv[1]);
    imshow("src", img);

    EntropyBasedContrastEnhancementTMO *my_entropy_based_contrast_enhancement_test = new EntropyBasedContrastEnhancementTMO();
    Mat out = my_entropy_based_contrast_enhancement_test->Run(img);
    imshow("out", out);
    waitKey(0);

    return 0;
}
