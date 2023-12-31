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

class MyOriAnisotropicTest{
    public:
        MyOriAnisotropicTest();
        ~MyOriAnisotropicTest();

		Mat Run(Mat src);

    private:
        float pm_g1(float value, float k);
        float pm_g2(float value, float k);
        float pm_g3(float value, float k);

		float Compute_K_Percentile(Mat img, int nbins);
		Mat IterAnisotropicTest(Mat src, float k);
};
