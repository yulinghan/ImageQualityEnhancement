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
#include <fstream> 

using namespace cv; 
using namespace std;

class RawUtils{
    public:
        RawUtils();
        ~RawUtils();

        Mat RawRead(string file_path, int height, int width, int type);
        Mat Bayer2Rggb(Mat bayer);
        Mat Rggb2Bayer(Mat rggb);
        Mat SubBlack(Mat rggb, int *black_level, int *white_level);
        Mat AddBlack(Mat rggb, int *black_level);
        Mat AddWBgain(Mat rggb, float *wb_gain);

        Mat CcmAdjust(Mat rgb, float (*ccm)[3]);
        Mat GammaAdjust(Mat ccm_rgb, float gamma);
        Mat GainAdjust(Mat rggb, float isp_gain);

        Mat Bgr2Mosaicking(Mat bgr);
};
