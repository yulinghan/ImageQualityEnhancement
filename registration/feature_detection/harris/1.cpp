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
#include "harris.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    imshow("src", src);

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    MyHarrisTest *my_harris_test = new MyHarrisTest();
    vector<Point> corners = my_harris_test->run(src_gray);

    cout << "corners:" << corners.size() << endl;

    Mat show_mat = my_harris_test->CornersShow(src, corners);
    imshow("show_mat", show_mat);

    waitKey(0);
	return 0;
}
