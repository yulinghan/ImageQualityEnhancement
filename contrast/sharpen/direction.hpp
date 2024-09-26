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

class MyDirectionTest{
    public:
        MyDirectionTest();
        ~MyDirectionTest();

		Mat Run(Mat src, int r, float scale);

	private:
		Mat GetDirectionEdge(Mat src);
		Mat GetAdjustMat(Mat src, Mat edge_mat, int r, float scale);
        short CalcKernel(uchar *data, char *kernel, int number);

	private:
		char kernel_kirsch[4][25] = {
			{
			 	 1,  1,  1,  1,  1,
			 	 1,  2,  2,  2,  1,
			 	 0,  0,  0,  0,  0,
             	-1, -2, -2, -2, -1, 
             	-1, -1, -1, -1, -1, 
			},
			{
			 	1,  1, 0, -1, -1,
				1,  2, 0, -2, -1,
				1,  2, 0, -2, -1, 
				1,  2, 0, -2, -1,
				1,  1, 0, -1, -1
			},
			{
				 0,  1,  1,  1, 1,
				-1,  0,  2,  2, 1,
				-1, -2,  0,  2, 1,
				-1, -2, -2,  0, 1,
				-1, -1, -1, -1, 0, 
			},
			{
				1,  1,  1,  1,  0,
				1,  2,  2,  0, -1,
				1,  2,  0, -2, -1,
				1,  0, -2, -2, -1,
				0, -1, -1, -1, -1, 
			}
		};
};
