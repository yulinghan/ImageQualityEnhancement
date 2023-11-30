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
#include "lbp.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    imshow("src", src);

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    MyLbpTest *my_lbp_test = new MyLbpTest();
    
    int type;
   
    //原始lbp
    type = 0;
    Mat ori_lbp = my_lbp_test->run(src_gray, type);
    imshow("ori_lbp", ori_lbp);

    //圆形lbp
    type = 1;
    Mat cir_lbp = my_lbp_test->run(src_gray, type);
    imshow("cir_lbp", cir_lbp);

    //旋转不变lbp
    type = 2;
    Mat r_lbp = my_lbp_test->run(src_gray, type);
    imshow("r_lbp", r_lbp);

    //等价模式lbp
    type = 3;
    Mat u_lbp = my_lbp_test->run(src_gray, type);
    imshow("cir_lbp", u_lbp);

    waitKey(0);
	return 0;
}
