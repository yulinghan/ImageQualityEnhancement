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

class MyMultiScaleSharpenTest{
    public:
        MyMultiScaleSharpenTest();
        ~MyMultiScaleSharpenTest();

		Mat Run(Mat src, int r,  float scale);

    private:
        int Sign(int x);
};
