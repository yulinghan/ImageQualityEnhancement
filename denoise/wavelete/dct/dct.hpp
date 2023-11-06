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

class MyDctTest{
    public:
        MyDctTest();
        ~MyDctTest();

		Mat dct_decompose(Mat src);
        Mat dct_recover(Mat src);

    private:
        float PI=3.1415926;
};
