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

class MyPyrContrastTest{
    public:
        MyPyrContrastTest();
        ~MyPyrContrastTest();

		Mat Run(Mat src);

	private:
        Mat PyrBuild(vector<Mat> pyr, int n_scales);
        vector<Mat> LaplacianPyramid(Mat img, int level);
};
