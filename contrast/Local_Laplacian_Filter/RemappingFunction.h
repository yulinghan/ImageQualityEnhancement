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

float EdgeRemap(float beta, float delta);
float DetailRemap(float delta, float alpha, float sigma_r);
float SmoothStep(float x_min, float x_max, float x);

void Evaluate(float value, float reference, float alpha, float beta, float sigma_r, float& output);
Mat Evaluate(Mat input, float reference, float alpha, float beta, float sigma_r);
