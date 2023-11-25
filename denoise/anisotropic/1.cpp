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
#include "ori_anisotropic.hpp"
#include "aos_anisotropic.hpp"

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

	MyOriAnisotropicTest *my_ori_anisotropic_test = new MyOriAnisotropicTest();
    Mat ori_anisotropic_blur = my_ori_anisotropic_test->Run(src);
    imshow("ori_anisotropic_blur", ori_anisotropic_blur);

	MyAosAnisotropicTest *my_aos_anisotropic_test = new MyAosAnisotropicTest();
    Mat aos_anisotropic_blur = my_aos_anisotropic_test->Run(src);
    imshow("aos_anisotropic_blur", aos_anisotropic_blur);

    Mat gauss_mat;
    GaussianBlur(src, gauss_mat, Size(15, 15), 0, 0);
    imshow("gauss_mat", gauss_mat);

    waitKey(0);

    return 0;
}
