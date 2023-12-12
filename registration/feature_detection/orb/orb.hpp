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

class MyOrbTest{
    public:
        MyOrbTest();
        ~MyOrbTest();

        void Run(Mat input);

    private:
        void runByImageBorder(vector<KeyPoint>& keypoints, Size imageSize, int borderSize);
        void computeKeyPoints(vector<Mat> src_pyr, vector<KeyPoint> &allKeypoints, int nfeatures,
                    int edgeThreshold, int patchSize, int fastThreshold, int border);
        void HarrisResponses(Mat src, vector<KeyPoint> &key_points, int blockSize, float HARRIS_K);
        vector<KeyPoint> DistanceChoice(vector<KeyPoint> allKeypoints, float minDistance);
        void ICAngles(vector<Mat> src_pyr, vector<KeyPoint> &pts, int patchSize, int border);
        Mat CornersShow(Mat src, vector<KeyPoint> corners);
};
