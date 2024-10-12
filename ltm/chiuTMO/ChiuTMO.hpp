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

class ChiuTMO{
    public:
        ChiuTMO();
        ~ChiuTMO();

		Mat Run(Mat src);

    private:
        Mat ChangeLuminance(Mat src, Mat new_l, Mat old_l);
        Mat ChiuGlare(Mat Ls, float k, float n, float w);
        Mat ClampAdjust(Mat S, Mat src_gray, float c_clamping);
};
