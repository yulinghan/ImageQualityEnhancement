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
#include "surf.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1], 0);
    resize(src, src, src.size()/4);
    imshow("src", src);

    MySurfTest *my_surf_test = new MySurfTest();
    my_surf_test->run(src);

    waitKey(0);
	return 0;
}
