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

class MyBriefTest{
    public:
        MyBriefTest();
        ~MyBriefTest();

        Mat Run(Mat input, vector<KeyPoint> corners);

    private:
        void runByImageBorder(std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize);
        int smoothedSum(Mat sum, KeyPoint pt, int y, int x);
        Mat pixelTests16(Mat sum, vector<KeyPoint> corners);

    private:
        int KERNEL_SIZE = 9;
        int PATCH_SIZE = 48;
};
