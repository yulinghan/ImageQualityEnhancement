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
#include <gsl/gsl_multimin.h>

using namespace cv;
using namespace std;

class MyConvPyrPoissonTrain{
    public:
        MyConvPyrPoissonTrain();
        ~MyConvPyrPoissonTrain();

		void Run(Mat src);
		void CalImageDivAndGrad(Mat src, Mat &div_g);
		Mat evalf(Mat cur_div_old, Mat h1, Mat h2, Mat g);
		void constructKernels(Mat w_img, Mat &h1, Mat &h2, Mat &g);

	public:
		int kernel_size_ = 5;
		Mat cur_i_, cur_div_;
};
