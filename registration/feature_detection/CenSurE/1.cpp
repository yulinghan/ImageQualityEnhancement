#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>  
#include <unistd.h>
#include "star.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    imshow("src_gray", src_gray);

    MyStarTest *my_star_test = new MyStarTest();
    Mat star = my_star_test->run(src_gray);
    imshow("star", star);

    waitKey(0);
	return 0;
}
