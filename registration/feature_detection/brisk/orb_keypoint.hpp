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

class MyOrbKeyPointTest{
    public:
        MyOrbKeyPointTest();
        ~MyOrbKeyPointTest();

        vector<KeyPoint> Run(Mat input, int fastThreshold);
        Mat CornersShow(Mat src, vector<KeyPoint> corners);
        int CornerScore(const uchar* ptr, const int pixel[], int threshold);
        void MakeOffsets(int pixel[25], int rowStride, int patternSize);

    private:
        void OrbKeyPoint(Mat img, vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression);
};
