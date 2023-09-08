#ifndef _MTBFramesRegistration__H_
#define _MTBFramesRegistration__H_

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

class MyClaheTest{
    public:
        MyClaheTest();
        ~MyClaheTest();

		Mat Run(Mat src, int step, float scale);

	private:
		vector<vector<float>> GetAdjustParam(Mat src, int width_block, int height_block, int step, float scale);
		Mat GetAdjustMat(Mat src, vector<vector<float>> hist_arr, int width_block, int height_block, int step);

};
#endif
