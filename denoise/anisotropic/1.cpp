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
#include "anisotropic.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

	MyAnisotropicTest *my_anisotropic_test = new MyAnisotropicTest();
    Mat anisotropic_blur = my_anisotropic_test->Run(src);
    imshow("anisotropic_blur", anisotropic_blur);

    Mat gauss_mat;
    GaussianBlur(src, gauss_mat, Size(15, 15), 0, 0);
    imshow("gauss_mat", gauss_mat);

    waitKey(0);

    return 0;
}
