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

class BruceExpoBlendTMO{
    public:
        BruceExpoBlendTMO();
        ~BruceExpoBlendTMO();

		Mat Run(Mat src);

    private:
        vector<Mat> CreateLDRStackFromHDR(Mat src);
        Mat GetResult(vector<Mat> ldr_arr, Mat src);
};
