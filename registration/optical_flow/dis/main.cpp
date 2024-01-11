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
    imwrite("src1.jpg", gray_src1);
    imwrite("src2.jpg", gray_src2);

    MyDis *my_dis_test = new MyDis();
    Mat flow = my_dis_test->run(gray_src1, gray_src2);

    vector<Mat> channels;
    split(flow, channels);
    for(int i=0; i<gray_src2.rows; i++) {
        for(int j=0; j<gray_src2.cols; j++) {
            channels[0].at<float>(i, j) += j;
            channels[1].at<float>(i, j) += i;
        }
    }

    Mat dst;
    remap(gray_src2, dst, channels[0], channels[1], cv::INTER_LINEAR, cv::BORDER_REFLECT);
    imshow("dst", dst);
    imwrite(argv[3], dst);

    waitKey(0);
	return 0;
}
