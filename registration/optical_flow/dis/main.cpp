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
#include "dis.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src1 = imread(argv[1]);
    Mat src2 = imread(argv[2]);
    
    resize(src1, src1, src1.size()/4);
    resize(src2, src2, src1.size());

    Mat gray_src1, gray_src2;
    cvtColor(src1, gray_src1, COLOR_BGR2GRAY);
    cvtColor(src2, gray_src2, COLOR_BGR2GRAY);

    MyDis *my_dis_test = new MyDis();
    Mat flow = my_dis_test->run(gray_src1, gray_src2);

    waitKey(0);
	return 0;
}
