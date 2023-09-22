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

class MyExposureFusionTest{
    public:
        MyExposureFusionTest();
        ~MyExposureFusionTest();

		Mat Run(vector<Mat> src_arr);

	private:
        Mat ContrastCalculate(Mat src);
        Mat SaturationCalculate(Mat src);
        Mat LightCalculate(Mat src);
        vector<Mat> WeightCalculate(vector<Mat> src_arr);

        vector<Mat> GaussianPyramid(Mat img, int level);
        vector<Mat> LaplacianPyramid(Mat img, int level);
        Mat PyrBuild(vector<Mat> pyr, int n_scales);
        vector<Mat> EdgeFusion(vector<vector<Mat>> pyr_i_arr, vector<vector<Mat>> pyr_w_arr, int n_scales);

        Mat PyrResult(vector<Mat> src_arr, vector<Mat> weight_arr, int n_scales);
        Mat PyrFusion(vector<Mat> src_arr, vector<Mat> weight_arr);
};
