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

class MyBiorTest{
    public:
        MyBiorTest();
        ~MyBiorTest();

		Mat bior_decompose(Mat src);
        Mat bior_recover(Mat src);

    private:
        void bior15_coef( vector<float> &lp1, vector<float> &hp1, vector<float> &lp2, vector<float> &hp2);

    private:
        float PI=3.1415926;
};
