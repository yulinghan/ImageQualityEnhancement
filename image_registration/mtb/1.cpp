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
#include "MTBFramesRegistration.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat input = imread(argv[1], 0);
    Mat ref   = imread(argv[2], 0);

    MyMtbTest *my_mtb_test = new MyMtbTest();
    Mat warped = my_mtb_test->Run(input, ref);

    imwrite(argv[3], warped);
	return 0;
}
