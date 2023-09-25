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

class MySefTest{
    public:
        MySefTest();
        ~MySefTest();

	    vector<Mat> Run(Mat src, float a, float b);

	private:
        float MedianCal(Mat src);
        void FrameParamCal(Mat src, float a, float b, float mid_value, int &Mp, int &Ns);
        vector<Mat> FusionMatCal(Mat src, float a, float b, int Mp, int Ns);
};
