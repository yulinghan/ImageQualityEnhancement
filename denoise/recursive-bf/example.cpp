#include <stdio.h>
#include <time.h>
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
#include "rbf.hpp"

using namespace std;
using namespace cv;


int main(int argc, char*argv[]) {
	int n = 100;
	float sigma_spatial = 5.0;
	float sigma_range = 15.0;

	Mat src = imread(argv[1], 0);
	int width = src.cols, height = src.rows, channel = src.channels();

	Mat out = Mat::zeros(src.size(), src.type());

	uchar* src_ptr = (uchar*)src.data;
	uchar* out_ptr = (uchar*)out.data;
	for (int i = 0; i < n; ++i) {
		recursive_bf(src_ptr, out_ptr, sigma_spatial, sigma_range, width, height, channel);
	}


	float * buffer = new float[(width * height* channel + width * height
			+ width * channel + width) * 2];

	for (int i = 0; i < n; ++i) {
		recursive_bf(src_ptr, out_ptr, sigma_spatial, sigma_range, width, height, channel, buffer);
	}

	imwrite(argv[2], out);

	return 0;
}
