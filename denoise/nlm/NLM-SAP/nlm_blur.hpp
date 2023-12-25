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

class MyNlmBlurTest{
    public:
        MyNlmBlurTest();
        ~MyNlmBlurTest();

		Mat Run(Mat src, float h, int halfKernelSize, int halfSearchSize);

    private:
        void CalLookupTable1(void);
        void CalLookupTable2(void);
        float MseBlock(Mat m1, Mat m2);
        Mat Nlm(Mat src, float h, int halfKernelSize, int halfSearchSize);

    private:
        float table1[256];
        uchar table2[256][256];
};
