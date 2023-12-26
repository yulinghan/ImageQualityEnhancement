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

class MyPullPushNlmTest{
    public:
        MyPullPushNlmTest();
        ~MyPullPushNlmTest();

		void PullNlm(Mat src, float h, int halfKernelSize, int halfSearchSize);

    private:
        float MseBlock(Mat m1, Mat m2);
		Mat DownFuse(Mat src, float h, int halfKernelSize, int halfSearchSize);
        Mat UpFuse(Mat src_f, Mat src_c, float h, int halfKernelSize, int halfSearchSize);

    private:
        vector<Mat> pull_src_arr;
        vector<Mat> pull_weight_p_arr;
        vector<Mat> push_weight_p_arr;
};
