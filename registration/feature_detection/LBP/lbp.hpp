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

class MyLbpTest{
    public:
        MyLbpTest();
        ~MyLbpTest();

        Mat run(Mat input, int type);

    private:
        Mat OriLbp(Mat img);
        Mat ELBP(Mat img, int radius, int neighbors);
        Mat RILBP(Mat img);
        Mat UniformLBP(Mat img);
};
