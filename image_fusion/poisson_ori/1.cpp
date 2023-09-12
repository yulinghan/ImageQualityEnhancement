#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "poissonori.hpp"

int main(int argc, char** argv) {
	Mat img1, img2;
	Mat in1 = imread(argv[1]);
	Mat in2 = imread(argv[2]);
	in1.convertTo(img1, CV_64FC3);
	in2.convertTo(img2, CV_64FC3);

	int posXinPic2 = 350;
	int posYinPic2 = 50;

	Rect rc = Rect(0, 0, in1.cols, in1.rows);
	MyPoissonOriTest *my_poisson_ori_test = new MyPoissonOriTest();
	Mat result = my_poisson_ori_test->Run(img1, img2, rc, posXinPic2, posYinPic2);
	result.convertTo(result, CV_8UC1);
	Rect rc2 = Rect(posXinPic2, posYinPic2, in1.cols, in1.rows);
	Mat roimat = in2(rc2);
	result.copyTo(roimat);

	imwrite(argv[3], in2);

	return 0;
}
