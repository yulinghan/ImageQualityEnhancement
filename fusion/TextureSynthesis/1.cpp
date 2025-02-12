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
#include "texture_synthesis.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], 0);
    imshow("src", src);

    MyTextureSynthesisTest *my_texture_synthesis_test = new MyTextureSynthesisTest();
        
	Mat out = my_texture_synthesis_test->Run(src);

    imshow("out", out);
    waitKey(0);

    return 0;
}
