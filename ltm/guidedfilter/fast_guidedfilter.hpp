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
#include <omp.h>

using namespace cv;
using namespace std;

class MyFastGuidedfilterTest{
    public:
        MyFastGuidedfilterTest();
        ~MyFastGuidedfilterTest();

		Mat Run(Mat I, Mat p, int r, float eps, int size);
};
