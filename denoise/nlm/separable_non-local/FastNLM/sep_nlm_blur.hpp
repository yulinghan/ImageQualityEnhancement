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

class MySepNlmBlurTest{
    public:
        MySepNlmBlurTest();
        ~MySepNlmBlurTest();

		Mat Run(Mat src, float h, int halfKernelSize, int halfSearchSize);

    private:
        float MseBlock(Mat m1, Mat m2);
        Mat Nlm(Mat src, float h, int halfKernelSize, int halfSearchSize);
        Mat CalF(Mat src, int S, int K, float beta);

    private:
        float table1[256];
        uchar table2[256][256];
};
