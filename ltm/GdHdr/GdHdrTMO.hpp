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

class GdHdrTMO{
    public:
        GdHdrTMO();
        ~GdHdrTMO();

		Mat Run(Mat src, float alpha, float beta);

    private:
        vector<Mat> BuildGaussianPy(Mat pImage);
        void CalculateGradient(cv::Mat& pImage, int level, cv::Mat& pGradX, cv::Mat& pGradY);
        void CalculateScaling(Mat& pGradMag, Mat& pScaling);
        void CalculateAttenuations(vector<Mat> pScalings, Mat& Attenuation);
        void CalculateDivergence(cv::Mat& Gx, cv::Mat& Gy, cv::Mat& divG);
        void CalculateAttenuatedGradient(cv::Mat& pImage, cv::Mat& phi, cv::Mat& pGradX, cv::Mat& pGradY);

        Mat FFTCalcu(Mat div_g);
        Mat ApplyToneMapping(Mat logLuma);

    private:
        float m_alpha;
        float m_beta;
};
