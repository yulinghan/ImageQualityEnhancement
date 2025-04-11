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
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;

#pragma once
#include <iostream>//cout

class CPchip
{
public:
	CPchip(vector<float> X, vector<float> Y, int n, double x);
	~CPchip();
	int FindIndex(vector<float> X, int n, double x);
	double ComputeDiff(vector<float> h, vector<float> delta, int n, int k);
	double getY() { return y; };

private:
	double y;
	double *h;
	double *delta;
};
