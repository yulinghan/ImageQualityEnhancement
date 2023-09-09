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
#include <vector>

using namespace cv;
using namespace std;

class MyPoissonFusionTest {
	public:
		MyPoissonFusionTest();
		~MyPoissonFusionTest();

		Mat run(vector<Mat> src_arr, vector<Mat> mask_arr);

	protected:
		void initVariables(Mat destination, vector<float> &filter_X, vector<float> &filter_Y);
		void computeDerivatives(vector<Mat> src_arr, vector<Mat> &patchGradientX, vector<Mat> &patchGradientY);
		Mat poisson(Mat destination, vector<Mat> mask_arr, vector<Mat> patchGradientX, vector<Mat> patchGradientY);
		void dst(const cv::Mat& src, cv::Mat& dest, bool invert = false);
		void solve(Mat &img, Mat& mod_diff, Mat &result);

		Mat poissonSolver(Mat img, Mat gxx, Mat gyy);
		void computeLaplacianX(Mat img, Mat &gxx);
		void computeLaplacianY(Mat img, Mat &gyy);
		void computeGradientX(Mat img, Mat &gx);
		void computeGradientY(Mat img, Mat &gy);

	private:
		Mat destinationGradientX, destinationGradientY;
};
