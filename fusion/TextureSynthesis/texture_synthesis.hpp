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

class MyTextureSynthesisTest{
    public:
        MyTextureSynthesisTest();
        ~MyTextureSynthesisTest();

		Mat Run(Mat src);

    private:
        vector<vector<Mat>> GetHaarPyr(Mat src, int level);
        Mat GetTextureSynthesis(vector<vector<Mat>> src_haar_pyr_arr, vector<vector<Mat>> noise_haar_pyr_arr, int max_value);
};
