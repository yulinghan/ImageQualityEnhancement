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
#include "nms.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    resize(src, src, Size(640, 480));
    imshow("src", src);

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    MyNmsTest *my_nms_test = new MyNmsTest();

    int maxCorners = 2000;
    vector<KeyPoint> corners_1 = my_nms_test->run(src_gray, maxCorners);
    cout << "corners_1:" << corners_1.size() << endl;

    Mat show_mat1 = my_nms_test->CornersShow(src, corners_1);
    imshow("show_mat1", show_mat1);

    maxCorners = 500;
    float minDistance = 11.0;
    vector<KeyPoint> corners_2;
    my_nms_test->DistanceChoice(minDistance, maxCorners, corners_2);
    cout << "corners_2:" << corners_2.size() << endl;
    Mat show_mat2 = my_nms_test->CornersShow(src, corners_2);
    imshow("show_mat2", show_mat2);

    maxCorners = 500;
    vector<KeyPoint> corners_3 = my_nms_test->ANMS(corners_1, maxCorners);
    cout << "corners_3:" << corners_3.size() << endl;
    Mat show_mat3 = my_nms_test->CornersShow(src, corners_3);
    imshow("show_mat3", show_mat3);

    maxCorners = 500;
    float tolerance = 0.1;
    vector<KeyPoint> corners_4 = my_nms_test->ssc(corners_1, maxCorners, tolerance, src.cols, src.rows);
    cout << "corners_4:" << corners_4.size() << endl;
    Mat show_mat4 = my_nms_test->CornersShow(src, corners_4);
    imshow("show_mat4", show_mat4);

    waitKey(0);
	return 0;
}
