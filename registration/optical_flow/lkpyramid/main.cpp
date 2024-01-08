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
#include "lkpyramid.hpp"

using namespace cv;
using namespace std;

bool rule(const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
    return p1.response > p2.response;
}

vector<KeyPoint> GetFastKeyPoints(Mat im) {
    Mat curMask = Mat(im.rows, im.cols, CV_8UC1, Scalar(255));
    vector<KeyPoint> key_points1, key_points2;

    auto detector = FastFeatureDetector::create(20, true);
    detector->detect(im, key_points1);

    KeyPointsFilter::retainBest(key_points1, 400);

    //sort features
    sort(key_points1.begin(), key_points1.end(), rule);

    for(int i=0; i<key_points1.size(); i++) {
        KeyPoint it = key_points1[i];

        //check is features are being too close
        if (curMask.at<uchar>(Point(it.pt.x, it.pt.y)) == 0) {
            continue;
        }

        //consider those weak features
        if (it.response < 20) {
            //variance of a patch
            float textureness;

            if(it.pt.x-2<0 || it.pt.y-2<0 || it.pt.x+2>im.cols-1 || it.pt.y+2>im.rows-1) {
                continue;
            }

            Mat patch = im(Rect(it.pt.x-2, it.pt.y-2, 5, 5));
            Scalar mean, stddev;
            meanStdDev(patch, mean, stddev);
            if(stddev.val[0] < 4) {
                continue;
            }
        }

        //runs to here means this feature will be kept
        rectangle(curMask, Point(int(it.pt.x-15/2+.5), int(it.pt.y-15/2+.5)), 
                        Point(int(it.pt.x+15/2+.5), int(it.pt.y+15/2+.5)), Scalar(0), -1);

        key_points2.push_back(it);
    }
    
    return key_points2;
}


int main(int argc, char* argv[]){
    Mat src1 = imread(argv[1]);
    Mat src2 = imread(argv[2]);
    
    resize(src1, src1, src1.size()/4);
    resize(src2, src2, src1.size());

    Mat gray_src1, gray_src2;
    cvtColor(src1, gray_src1, COLOR_BGR2GRAY);
    cvtColor(src2, gray_src2, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints1 = GetFastKeyPoints(gray_src1);
    cout << "keypoints1:" << keypoints1.size() << endl;

    Mat show_mat1;
    drawKeypoints(src1, keypoints1, show_mat1, Scalar::all(255));
    imshow("show_mat1", show_mat1);

    LKPyramid *my_lkpyramid_test = new LKPyramid();
    vector<KeyPoint> keypoints2 = my_lkpyramid_test->run(gray_src1, gray_src2, keypoints1);
    cout << "keypoints2:" << keypoints2.size() << endl;

    Mat show_mat2;
    drawKeypoints(src2, keypoints2, show_mat2, Scalar::all(255));
    imshow("show_mat2", show_mat2);

    waitKey(0);
	return 0;
}
