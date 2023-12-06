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

class MyStarTest{
    public:
        MyStarTest();
        ~MyStarTest();

        Mat run(Mat input);

    private:
        int StarDetectorComputeResponses(Mat img, Mat &responses, Mat &sizes, int maxSize);
        void computeIntegralImages(Mat& matI, Mat& matS, Mat& matT, Mat& _FT);

        void StarDetectorSuppressNonmax(Mat& responses, Mat& sizes,
                            vector<KeyPoint>& keypoints, int border,
                            int responseThreshold, int lineThresholdProjected,
                            int lineThresholdBinarized, int suppressNonmaxSize);

        bool StarDetectorSuppressLines(const Mat& responses, const Mat& sizes, Point pt,
                                       int lineThresholdProjected, int lineThresholdBinarized);

        Mat CornersShow(Mat src, vector<KeyPoint> corners);
};
