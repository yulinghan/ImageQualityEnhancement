#include "poissonori.hpp"

MyPoissonOriTest::MyPoissonOriTest() {
}

MyPoissonOriTest::~MyPoissonOriTest() {
}

int MyPoissonOriTest::GetLabel(int i, int j, int height, int width) {
	return i * width + j;
}

Mat MyPoissonOriTest::GetA(int height, int width) {
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
			int label = GetLabel(i, j, height, width);
			if(M.at<double>(i, j) == 0){
				if(i == 0)  A.at<double>(GetLabel(i + 1, j, height, width), label) = 1;
				else if(i == height - 1)   A.at<double>(GetLabel(i - 1, j, height, width), label) = 1;
				if(j == 0)  A.at<double>(GetLabel(i, j + 1, height, width), label) = 1;
				else if(j == width - 1)   A.at<double>(GetLabel(i, j - 1, height, width), label) = 1;
			}else if(M.at<double>(i, j) == 1){
				if(i == 0){
					A.at<double>(GetLabel(i + 1, j, height, width), label) = 1;
					A.at<double>(GetLabel(i, j - 1, height, width), label) = 1;
					A.at<double>(GetLabel(i, j + 1, height, width), label) = 1;
				}else if(i == height - 1){
					A.at<double>(GetLabel(i - 1, j, height, width), label) = 1;
					A.at<double>(GetLabel(i, j - 1, height, width), label) = 1;
					A.at<double>(GetLabel(i, j + 1, height, width), label) = 1;
				}
				if(j == 0){
					A.at<double>(GetLabel(i, j + 1, height, width), label) = 1;
					A.at<double>(GetLabel(i - 1, j, height, width), label) = 1;
					A.at<double>(GetLabel(i + 1, j, height, width), label) = 1;
				}else if(j == width - 1){
					A.at<double>(GetLabel(i, j - 1, height, width), label) = 1;
					A.at<double>(GetLabel(i - 1, j, height, width), label) = 1;
					A.at<double>(GetLabel(i + 1, j, height, width), label) = 1;
				}
			}else{
				A.at<double>(GetLabel(i, j - 1, height, width), label) = 1;
				A.at<double>(GetLabel(i, j + 1, height, width), label) = 1;
				A.at<double>(GetLabel(i - 1, j, height, width), label) = 1;
				A.at<double>(GetLabel(i + 1, j, height, width), label) = 1;
			}
		}
	}
	return A;
}

Mat MyPoissonOriTest::GetLaplacian(){
    Mat laplacian = Mat::zeros(3, 3, CV_64FC1);
    laplacian.at<double>(0, 1) = 1.0;
    laplacian.at<double>(1, 0) = 1.0;
    laplacian.at<double>(1, 2) = 1.0;
    laplacian.at<double>(2, 1) = 1.0;
    laplacian.at<double>(1, 1) = -4.0; 
    return laplacian;
}

Mat MyPoissonOriTest::GetB(Mat img1, Mat img2, int posX, int posY, Rect ROI){
    Mat Lap;
    filter2D(img1, Lap, -1, GetLaplacian());
    int roiheight = ROI.height;
    int roiwidth = ROI.width;
    Mat B = Mat::zeros(roiheight * roiwidth, 1, CV_64FC1);
    for(int i=0; i<roiheight; i++){
        for(int j=0; j<roiwidth; j++){
            double temp = 0.0;
            temp += Lap.at<double>(i + ROI.y, j + ROI.x);
            if(i == 0)              temp -= img2.at<double>(i - 1 + posY, j + posX);
            if(i == roiheight - 1)  temp -= img2.at<double>(i + 1 + posY, j + posX);
            if(j == 0)              temp -= img2.at<double>(i + posY, j - 1 + posX);
            if(j == roiwidth - 1)   temp -= img2.at<double>(i + posY, j + 1 + posX);
            B.at<double>(GetLabel(i, j, roiheight, roiwidth), 0) = temp;
        }
    }
    return B;
}

Mat MyPoissonOriTest::GetResult(Mat A, Mat B, Rect ROI){
    Mat result;
    solve(A, B, result);
    result = result.reshape(0, ROI.height);
    return  result;
}

Mat MyPoissonOriTest::Run(Mat img1, Mat img2, Rect ROI, int posX, int posY) {
	int roiheight = ROI.height;
    int roiwidth = ROI.width;
    Mat A = GetA(roiheight, roiwidth);                                                                                                                                                                     

    vector<Mat> rgb1, rgb2;
    split(img1, rgb1);
    split(img2, rgb2);

    vector<Mat> result;
    Mat merged, res, Br, Bg, Bb;

    Br = GetB(rgb1[0], rgb2[0], posX, posY, ROI);
    res = GetResult(A, Br, ROI);
    result.push_back(res);

    Bg = GetB(rgb1[1], rgb2[1], posX, posY, ROI);
    res = GetResult(A, Bg, ROI);
    result.push_back(res);

    Bb = GetB(rgb1[2], rgb2[2], posX, posY, ROI);
    res = GetResult(A, Bb, ROI);
    result.push_back(res);

    merge(result,merged);
    return merged; 
}
