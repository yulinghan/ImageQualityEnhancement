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
                                                                                                                                                                                                           
using namespace cv; 
using namespace std;

class LKPyramid{
    public:
        LKPyramid();
        ~LKPyramid();

        vector<KeyPoint> run(Mat src1, Mat src2, vector<KeyPoint> prev_key_points);

    private:
        vector<KeyPoint> MyCalcOpticalFlowPyrLK(vector<Mat> prev_pyr, vector<Mat> next_pyr, vector<KeyPoint> prev_points,
                                vector<uchar> status, vector<float> error, Size winSize, int maxLevel, TermCriteria criteria);


    private:
        double minEigThreshold_ = 1e-4;
};
