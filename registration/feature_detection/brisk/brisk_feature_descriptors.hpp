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

struct BriskPatternPoint{
	float x;         // x coordinate relative to center
	float y;         // x coordinate relative to center
	float sigma;     // Gaussian smoothing sigma
};

struct BriskShortPair{
	unsigned int i;  // index of the first pattern point
	unsigned int j;  // index of other pattern point
};

struct BriskLongPair{
	unsigned int i;  // index of the first pattern point
	unsigned int j;  // index of other pattern point
	int weighted_dx; // 1024.0/dx
	int weighted_dy; // 1024.0/dy
};

class MyBriskFeatureDescriptorsTest{
    public:
        MyBriskFeatureDescriptorsTest(float patternScale);
        ~MyBriskFeatureDescriptorsTest();

        Mat run(Mat input, vector<KeyPoint> key_points);

    private:
        void GenerateKernel(vector<float> &radiusList, vector<int> &numberList, float dMax, float dMin);
        int smoothedIntensity(Mat& image, Mat& integral, float key_x, float key_y, int scale, int rot, int point);

    public:
        bool rotationInvariance;
        bool scaleInvariance;

    private:
        int strings_;
        int points_;
        int n_rot_ = 1024;
        float dMax_;
		float dMin_;
        int scales_ = 64;
        float scalerange_ = 30.f;
        float* scaleList_;
        int* sizeList_;

        BriskPatternPoint* patternPoints_;

        BriskShortPair* shortPairs_; 		// d<_dMax
		BriskLongPair* longPairs_; 			// d>_dMin
		int noShortPairs_; 		    // number of shortParis
		int noLongPairs_; 			// number of longParis

		float basicSize_ = 12.0f;
};
