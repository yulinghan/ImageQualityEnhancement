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
#include "orb.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    imshow("src", src_gray);

    MyOrbTest *my_orb_test = new MyOrbTest();
    my_orb_test->Run(src_gray);

    waitKey(0);
	return 0;
}
