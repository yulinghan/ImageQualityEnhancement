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

class MyAkazeKeyScaleSpaceTest{
    public:
        MyAkazeKeyScaleSpaceTest();
        ~MyAkazeKeyScaleSpaceTest();

        vector<vector<Mat>> run(Mat src);

    private:
        int GetGaussianKernelSize(float sigma);
        float Compute_K_Percentile(Mat img, int nbins);
        Mat weickert_diffusivity(Mat Lx, Mat Ly, float k);
        int fed_tau_by_process_time(float T, int M, float tau_max, vector<float>& tau);
        void nld_step_scalar_one_lane(Mat Lt, Mat Lf, Mat& Lstep, float step_size);
        vector<vector<Mat>> create_nonlinear_scale_space(Mat img);
        bool fed_is_prime_internal(int& number);

};
