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

class MyAosAnisotropicTest{
    public:
        MyAosAnisotropicTest();
        ~MyAosAnisotropicTest();

		Mat Run(Mat src);

    private:
        Mat pm_g1(Mat Lx, Mat Ly, float k);
        Mat pm_g2(Mat Lx, Mat Ly, float k);
        Mat pm_g3(Mat Lx, Mat Ly, float k);
		
		float Compute_K_Percentile(Mat Lx, Mat Ly, int nbins);

		Mat AosStepScalar(Mat Ldprev, Mat c, float stepsize);
		Mat AosColumns(Mat Ldprev, Mat c, float stepsize);
		Mat AosRows(Mat Ldprev, Mat c, float stepsize);
		Mat Thomas(Mat a, Mat b, Mat Ld);
};
