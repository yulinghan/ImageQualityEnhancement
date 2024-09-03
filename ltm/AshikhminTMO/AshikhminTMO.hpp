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

class AshikhminTMO{
    public:
        AshikhminTMO(float gamma, int ashikhmin_smax);
        ~AshikhminTMO();

		Mat Run(Mat src);

    private:
        void log_(const Mat& src, Mat& dst);
        Mat linear(Mat src, float gamma);
        void AshikhminFiltering(Mat src_gray, Mat &L, Mat &Ldetail);

    private:
        float m_gamma;
        int m_ashikhmin_smax;
};
