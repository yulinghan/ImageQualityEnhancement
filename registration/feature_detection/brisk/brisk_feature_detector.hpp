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

class MyBriskFeatureDetectorTest{
    public:
        MyBriskFeatureDetectorTest();
        ~MyBriskFeatureDetectorTest();

        vector<KeyPoint> run(Mat input, int threshold, int octaves);

    private:
        vector<Mat> ConstructPyramid(Mat src, int layers);
        vector<KeyPoint> GetKeyPoints(vector<Mat> src_pyr, int threshold, int layers);
        float GetScoreMaxAbove(vector<Mat> src_pyr, int pixel[25], int layer, int x_layer, int y_layer, int threshold, int center,
                                    bool& ismax, float& dx, float& dy);
        float GetScoreMaxBelow(vector<Mat> src_pyr, int pixel[25], int layer, int x_layer, int y_layer, int threshold, int center,
                                    bool& ismax, float& dx, float& dy);
        float refine1D(const float s_05, const float s0, const float s05, float& max);
        float refine1D_1(const float s_05, const float s0, const float s05, float& max);
        int GetScore(Mat src, int pixel[25], int threshold, int x, int y);
        float subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2,
                                    const int s_1_0, const int s_1_1, const int s_1_2,
                                    const int s_2_0, const int s_2_1, const int s_2_2,
                                    float& delta_x, float& delta_y);
};
