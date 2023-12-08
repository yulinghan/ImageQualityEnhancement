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
#include "fast_keypoint.hpp"
#include "brief_desc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    imshow("src", src);

    MyFastTest *my_fast_test = new MyFastTest();
    vector<KeyPoint> corners = my_fast_test->Run(src_gray);
    cout << "corners:" << corners.size() << endl;

    Mat fast_mat = my_fast_test->CornersShow(src, corners);
    imshow("fast_mat", fast_mat);

    MyBriefTest *my_brief_test = new MyBriefTest();
    Mat descriptors = my_brief_test->Run(src_gray, corners);

    cout << "descriptors:" << descriptors << endl;

    waitKey(0);
	return 0;
}
