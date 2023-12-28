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
#include "akaze_scale_space.hpp"
#include "akaze_key_points.hpp"
#include "akaze_desc.hpp"

using namespace cv;
using namespace std;

#define DEBUG 1

int main(int argc, char* argv[]){
    Mat src = imread(argv[1]);
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    imshow("src_gray", src_gray);

    MyAkazeKeyScaleSpaceTest *my_akaze_scale_space_test = new MyAkazeKeyScaleSpaceTest();
    vector<vector<Mat>> scale_space_arr = my_akaze_scale_space_test->run(src_gray);
#if DEBUG
    for(int i=0; i<scale_space_arr.size(); i++) {
        for(int j=0; j<scale_space_arr[i].size(); j++) {
            imshow(format("%d_%d", i, j), scale_space_arr[i][j]);
        }
    }
#endif

    MyAkazeKeyPointsTest *my_akaze_key_points_test = new MyAkazeKeyPointsTest();
    vector<KeyPoint> key_points = my_akaze_key_points_test->run(scale_space_arr);

#if DEBUG 
    Mat key_point_show = my_akaze_key_points_test->DispKeyPoint(src_gray, key_points);
    imshow("key_point_show", key_point_show);
    cout << "!!!! key_points:" << key_points.size() << endl;
#endif

    MyAkazeDescTest *my_akaze_desc_test = new MyAkazeDescTest();
    Mat desc = my_akaze_desc_test->run(scale_space_arr, key_points);

#if DEBUG
    for(int i=0; i<desc.rows; i++) {
        cout << "key_point[" << i << "]:" << endl;
        for(int j=0; j<desc.cols; j++) {
            cout << (int)desc.at<uchar>(i, j) << ", ";
        }
        cout << endl;
    }

#endif

    waitKey(0);
	return 0;
}
