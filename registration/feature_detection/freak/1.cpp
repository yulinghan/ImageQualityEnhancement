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
#include "shi_tomasi.hpp"
#include "freak_descriptors.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    imshow("src", src_gray);

    MyShiTomasiTest *my_shi_tomasi_test = new MyShiTomasiTest();
    vector<Point> corners = my_shi_tomasi_test->run(src_gray);
    cout << "corners:" << corners.size() << endl;
    Mat show_mat = my_shi_tomasi_test->CornersShow(src, corners);
    imshow("show_mat", show_mat);

    MyFreakDescriptorsTest *my_freak_descriptors_test = new MyFreakDescriptorsTest();
    Mat descriptors = my_freak_descriptors_test->run(src_gray, corners);
    cout << "descriptors:" << descriptors.size() << endl;

    waitKey(0);
	return 0;
}
