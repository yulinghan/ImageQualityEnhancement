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
#include <opencv2/ximgproc.hpp>
#include "poisson_fft.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src1 = imread(argv[1]);
    Mat src2 = imread(argv[2]);
    Mat mask = imread(argv[3], 0);

	Mat tmp_y;
    cvtColor(src1, tmp_y, COLOR_BGR2GRAY);
	ximgproc::guidedFilter(tmp_y, mask, mask, 7, 500, -1);

	vector<Mat> channels1, channels2;
	split(src1, channels1);
	split(src2, channels2);

	for(int i=0; i<src1.channels(); i++) {
		MyPoissonFusionTest *my_poisson_fusion_test = new MyPoissonFusionTest();

		vector<Mat> src_arr, mask_arr;
		src_arr.push_back(channels1[i]);
		src_arr.push_back(channels2[i]);
		mask_arr.push_back(255-mask);
		mask_arr.push_back(mask);
		channels1[i] = my_poisson_fusion_test->run(src_arr, mask_arr);
	}

	Mat out;
	merge(channels1, out);
	out.convertTo(out, CV_8U);
	imwrite(argv[4], out);

    return 0;
}
