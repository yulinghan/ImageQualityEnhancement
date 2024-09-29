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

class EntropyBasedContrastEnhancementTMO{
    public:
        EntropyBasedContrastEnhancementTMO();
        ~EntropyBasedContrastEnhancementTMO();

		Mat Run(Mat src);

    private:
        void init(Mat src);
        vector<vector<float>> spatialHistgram(Mat m_img_gray);
        void spatialEntropy(vector<vector<float>> patch_histogram_arr, vector<float> &f_cdf, vector<float> &s_k_arr);
        Mat mapping(vector<float> f_cdf, float yd, float yu, Mat src);
        Mat domainCoefWeight(Mat src, vector<float> s_k_arr, float gamma);
        Mat DomainEnhance(Mat src, Mat weight_mat);

    private:
        Mat m_img_gray, m_img_hsv, m_img_bgr;
        float m_eps = 1e-6;
        int m_level = 256;
};
