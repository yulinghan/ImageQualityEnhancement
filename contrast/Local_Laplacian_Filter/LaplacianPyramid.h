#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

int GetLevelCount(int rows, int cols, int desired_base_size);
vector<Mat> GaussianPyramid(Mat img, int level, int row_start, int row_end, int col_start, int col_end);
vector<int> GetLevelSize(int level, vector<int> base_subwindow);
Mat PopulateTopLevel(Mat src1, int KRows, int KCols, int row_offset, int col_offset);
Mat Expand(Mat input, int out_rows, int out_cols, int row_start, int row_end, int col_start, int col_end);
vector<Mat> LaplacianPyramid(Mat img, int level, int row_start, int row_end, int col_start, int col_end);
Mat LapReconstruct(vector<Mat> output);
