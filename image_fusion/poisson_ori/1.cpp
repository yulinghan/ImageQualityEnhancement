#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "vector"
#include "time.h"

#define elif else if
#define ATD at<double>
#define vector vector<Mat>

using namespace cv;
using namespace std;

//calculate horizontal gradient, img(i,j+1) - img(i,j)
Mat getGradientXp(Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	Mat cat = repeat(img, 1, 2);
	Rect roi = Rect(1, 0, width, height);
	Mat roimat = cat(roi);
	return roimat - img;
}


//calculate vertical gradient, img(i+1,j) - img(i,j)
Mat getGradientYp(Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	Mat cat = repeat(img, 2, 1);
	Rect roi = Rect(0, 1, width, height);
	Mat roimat = cat(roi);
	return roimat - img;
}

//calculate horizontal gradient, img(i,j-1) - img(i,j)
Mat getGradientXn(Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	Mat cat = repeat(img, 1, 2);
	Rect roi = Rect(width-1, 0, width, height);
	Mat roimat = cat(roi);
	return roimat - img;
}
//calculate vertical gradient, img(i-1,j) - img(i,j)
Mat getGradientYn(Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	Mat cat = repeat(img, 2, 1);
	Rect roi = Rect(0, height-1, width, height);
	Mat roimat = cat(roi);
	return roimat - img;
}

int getLabel(int i, int j, int height, int width)
{
	return i * width + j;
}

//get Matrix A.
Mat getA(int height, int width)
{
	Mat A = Mat::eye(height*width, height*width, CV_64FC1);
	A *= -4;
	Mat M = Mat::zeros(height, width, CV_64FC1);
	Mat temp = Mat::ones(height, width - 2, CV_64FC1);
	Rect roi = Rect(1, 0, width - 2, height);
	Mat roimat = M(roi);
	temp.copyTo(roimat);
	temp = Mat::ones(height - 2, width, CV_64FC1);
	roi = Rect(0, 1, width, height - 2);
	roimat = M(roi);
	temp.copyTo(roimat);
	temp = Mat::ones(height - 2, width - 2, CV_64FC1);
	temp *= 2;
	roi = Rect(1, 1, width - 2, height - 2);
	roimat = M(roi);
	temp.copyTo(roimat);
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			int label = getLabel(i, j, height, width);
			if(M.ATD(i, j) == 0){
				if(i == 0)  A.ATD(getLabel(i + 1, j, height, width), label) = 1;
				elif(i == height - 1)   A.ATD(getLabel(i - 1, j, height, width), label) = 1;
				if(j == 0)  A.ATD(getLabel(i, j + 1, height, width), label) = 1;
				elif(j == width - 1)   A.ATD(getLabel(i, j - 1, height, width), label) = 1;
			}elif(M.ATD(i, j) == 1){
				if(i == 0){
					A.ATD(getLabel(i + 1, j, height, width), label) = 1;
					A.ATD(getLabel(i, j - 1, height, width), label) = 1;
					A.ATD(getLabel(i, j + 1, height, width), label) = 1;
				}elif(i == height - 1){
					A.ATD(getLabel(i - 1, j, height, width), label) = 1;
					A.ATD(getLabel(i, j - 1, height, width), label) = 1;
					A.ATD(getLabel(i, j + 1, height, width), label) = 1;
				}
				if(j == 0){
					A.ATD(getLabel(i, j + 1, height, width), label) = 1;
					A.ATD(getLabel(i - 1, j, height, width), label) = 1;
					A.ATD(getLabel(i + 1, j, height, width), label) = 1;
				}elif(j == width - 1){
					A.ATD(getLabel(i, j - 1, height, width), label) = 1;
					A.ATD(getLabel(i - 1, j, height, width), label) = 1;
					A.ATD(getLabel(i + 1, j, height, width), label) = 1;
				}
			}else{
				A.ATD(getLabel(i, j - 1, height, width), label) = 1;
				A.ATD(getLabel(i, j + 1, height, width), label) = 1;
				A.ATD(getLabel(i - 1, j, height, width), label) = 1;
				A.ATD(getLabel(i + 1, j, height, width), label) = 1;
			}
		}
	}
	return A;
}

// Get the following Laplacian matrix
// 0  1  0
// 1 -4  1
// 0  1  0
Mat
getLaplacian(){
	Mat laplacian = Mat::zeros(3, 3, CV_64FC1);
	laplacian.ATD(0, 1) = 1.0;
	laplacian.ATD(1, 0) = 1.0;
	laplacian.ATD(1, 2) = 1.0;
	laplacian.ATD(2, 1) = 1.0;
	laplacian.ATD(1, 1) = -4.0; 
	return laplacian;
}
// Calculate b
// using convolution.
Mat getB1(Mat &img1, Mat &img2, int posX, int posY, Rect ROI){
	Mat Lap;
	filter2D(img1, Lap, -1, getLaplacian());
	int roiheight = ROI.height;
	int roiwidth = ROI.width;
	Mat B = Mat::zeros(roiheight * roiwidth, 1, CV_64FC1);
	for(int i=0; i<roiheight; i++){
		for(int j=0; j<roiwidth; j++){
			double temp = 0.0;
			temp += Lap.ATD(i + ROI.y, j + ROI.x);
			if(i == 0)              temp -= img2.ATD(i - 1 + posY, j + posX);
			if(i == roiheight - 1)  temp -= img2.ATD(i + 1 + posY, j + posX);
			if(j == 0)              temp -= img2.ATD(i + posY, j - 1 + posX);
			if(j == roiwidth - 1)   temp -= img2.ATD(i + posY, j + 1 + posX);
			B.ATD(getLabel(i, j, roiheight, roiwidth), 0) = temp;
		}
	}
	return B;
}

// Calculate b
// using getGradient functions.
Mat getB2(Mat &img1, Mat &img2, int posX, int posY, Rect ROI){
	Mat grad = getGradientXp(img1) + getGradientYp(img1) + getGradientXn(img1) + getGradientYn(img1);
	int roiheight = ROI.height;
	int roiwidth = ROI.width;
	Mat B = Mat::zeros(roiheight * roiwidth, 1, CV_64FC1);
	for(int i=0; i<roiheight; i++){
		for(int j=0; j<roiwidth; j++){
			double temp = 0.0;
			temp += grad.ATD(i + ROI.y, j + ROI.x);
			if(i == 0)              temp -= img2.ATD(i - 1 + posY, j + posX);
			if(i == roiheight - 1)  temp -= img2.ATD(i + 1 + posY, j + posX);
			if(j == 0)              temp -= img2.ATD(i + posY, j - 1 + posX);
			if(j == roiwidth - 1)   temp -= img2.ATD(i + posY, j + 1 + posX);
			B.ATD(getLabel(i, j, roiheight, roiwidth), 0) = temp;
		}
	}
	return B;
}

// Solve equation and reshape it back to the right height and width.
Mat getResult(Mat &A, Mat &B, Rect &ROI){
	Mat result;
	solve(A, B, result);
	result = result.reshape(0, ROI.height);
	return  result;
}

// img1: 3-channel image, we wanna move something in it into img2.
// img2: 3-channel image, dst image.
// ROI: the position and size of the block we want to move in img1.
// posX, posY: where we want to move the block to in img2
Mat
poisson_blending(Mat &img1, Mat &img2, Rect ROI, int posX, int posY){

	int roiheight = ROI.height;
	int roiwidth = ROI.width;
	Mat A = getA(roiheight, roiwidth);

	// we must do the poisson blending to each channel.
	vector rgb1;
	split(img1, rgb1);
	vector rgb2;
	split(img2, rgb2);

	vector result;
	Mat merged, res, Br, Bg, Bb;
	// For calculating B, you can use either getB1() or getB2()
	Br = getB2(rgb1[0], rgb2[0], posX, posY, ROI);
	//Br = getB2(rgb1[0], rgb2[0], posX, posY, ROI);
	res = getResult(A, Br, ROI);
	result.push_back(res);
	cout<<"R channel finished..."<<endl;
	Bg = getB2(rgb1[1], rgb2[1], posX, posY, ROI);
	//Bg = getB2(rgb1[1], rgb2[1], posX, posY, ROI);
	res = getResult(A, Bg, ROI);
	result.push_back(res);
	cout<<"G channel finished..."<<endl;
	Bb = getB2(rgb1[2], rgb2[2], posX, posY, ROI);
	//Bb = getB2(rgb1[2], rgb2[2], posX, posY, ROI);
	res = getResult(A, Bb, ROI);
	result.push_back(res);
	cout<<"B channel finished..."<<endl;

	// merge the 3 gray images into a 3-channel image 
	merge(result,merged);
	return merged; 
}

int main(int argc, char** argv) {
	Mat img1, img2;
	Mat in1 = imread(argv[1]);
	Mat in2 = imread(argv[2]);
	in1.convertTo(img1, CV_64FC3);
	in2.convertTo(img2, CV_64FC3);

	int posXinPic2 = 350;
	int posYinPic2 = 50;

	Rect rc = Rect(0, 0, in1.cols, in1.rows);
	Mat result = poisson_blending(img1, img2, rc, posXinPic2, posYinPic2);
	result.convertTo(result, CV_8UC1);
	Rect rc2 = Rect(posXinPic2, posYinPic2, in1.cols, in1.rows);
	Mat roimat = in2(rc2);
	result.copyTo(roimat);

	imwrite(argv[3], in2);

	return 0;
}
