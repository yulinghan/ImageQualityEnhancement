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
#include "surf_keypoint.hpp"
#include "surf_desc.hpp"
#include "common.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat bgr_src = imread(argv[1]);
    imshow("src", bgr_src);
    
    Mat src_gray;
    bgr_src.convertTo(src_gray, COLOR_BGR2GRAY);
    src_gray.convertTo(src_gray, CV_32FC1, 1/255.0);
    Mat integ_mat = Integral(src_gray);

    MySurfKeyPointTest *my_surf_key_point_test = new MySurfKeyPointTest();
    vector<MyKeyPoint> key_point_vec = my_surf_key_point_test->run(integ_mat);

#if 1
    Mat disp_key_point = my_surf_key_point_test->DispKeyPoint(bgr_src, key_point_vec);
    imshow("disp_key_point", disp_key_point);
#endif

    MySurfDescTest *my_surf_desc_test = new MySurfDescTest();
    my_surf_desc_test->run(integ_mat, key_point_vec);

#if 1
    Mat disp_desc = my_surf_desc_test->DispDesc(bgr_src, key_point_vec);
    imshow("disp_desc", disp_desc);

#endif
    waitKey(0);
	return 0;
}
