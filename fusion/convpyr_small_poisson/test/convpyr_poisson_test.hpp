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
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;

class MyConvPyrPoissonTest{
    public:
        MyConvPyrPoissonTest();
        ~MyConvPyrPoissonTest();

		Mat Run(Mat src1, Mat src2, Mat mask);
	
	private:
		Mat CalImageDivAndGrad(Mat src1, Mat src2, Mat mask1, Mat mask2);
		Mat evalf(Mat cur_div_old, Mat h1, Mat h2, Mat g);
		void constructKernels(Mat w_img, Mat &h1, Mat &h2, Mat &g);
};
