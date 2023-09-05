#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include "segment/image.h"
#include "segment/misc.h"
#include "segment/pnmfile.h"
#include "segment/segment-image.h"

using namespace cv;
using namespace std;


#define REGIONNUMBER   7

double myVb[REGIONNUMBER], myVh[REGIONNUMBER], regionSize[REGIONNUMBER];
double myEi[REGIONNUMBER], myPij[REGIONNUMBER][REGIONNUMBER];
double myQs, myQh;

int relaxDij[REGIONNUMBER][2];
Mat histogramMat[REGIONNUMBER];
int arrRegion[REGIONNUMBER];

/******************区域分割******************************/
Mat mySegment(Mat &src, int &num_ccs){
	float sigma = 0.15;
	float k = 300;
	int min_size = 100;

	Mat out;

	image<uchar> *input = loadMat(src);
	image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs);	
	out = getMat(seg);

	return out;
}

/******************相同亮度区域层合并******************************/
Mat lightSameSeg(Mat &segMat, Mat &srcMat, int num_ccs){
	int i, j, m;
	double lightArr[num_ccs];
	double pixelTmp, pixelAvg[num_ccs];
	Mat out = Mat::zeros(segMat.size(), CV_8UC1);
	for(i=0; i<num_ccs; i++){
		lightArr[i] = 0;
		pixelAvg[i] = 0;
	}

	double srcMaxPixel, srcMinPixel;
	minMaxIdx(srcMat, &srcMinPixel, &srcMaxPixel);

	for(i=0; i<segMat.rows; i++){
		uchar *ptr1 = segMat.ptr<uchar>(i);
		uchar *ptr2 = srcMat.ptr<uchar>(i);
		for(j=0; j<segMat.cols; j++){
			m = ptr1[j*3+1] + ptr1[j*3+2]*255;
			pixelAvg[m]  = (pixelAvg[m] + (ptr2[j] - srcMinPixel) / (srcMaxPixel - srcMinPixel + 0.00000001)) / 2;
		}
	}
	
	for(i=0; i<num_ccs; i++){
		int tmp = pixelAvg[i] * 10;
		pixelAvg[i] = double(tmp) / 10;
	}

	for(i=0; i<segMat.rows; i++){
		uchar *ptr1 = segMat.ptr<uchar>(i);
		uchar *ptr2 = out.ptr<uchar>(i);
		for(j=0; j<segMat.cols; j++){
			m = ptr1[j*3+1] + ptr1[j*3+2]*255;
			int tmp;
			if(pixelAvg[m] >= 0.5 && pixelAvg[m] <= 0.7){
				tmp = 7;
			}else if(pixelAvg[m] > 0.7){
				tmp = 8;
			}else{
				tmp = (int)(pixelAvg[m] * 10);
			}
			ptr2[j] = tmp;
		}
	}
	return out;
}

/***********************计算分割区域细节量和面积权重******************/
void myGammaDetile(Mat &lightSegMat, Mat &srcMat){
	double tmpPixelAll[REGIONNUMBER], tmpPixelB[REGIONNUMBER], tmpPixelH[REGIONNUMBER];
	int regionsize[REGIONNUMBER];

	Mat tmpMat    = Mat::zeros(srcMat.size(), CV_8UC1);
	Mat srcMatGaL = Mat::zeros(srcMat.size(), CV_8UC1);
	Mat srcMatGaH = Mat::zeros(srcMat.size(), CV_8UC1);

	Mat cannyMatAll = Mat::zeros(srcMat.size(), CV_8UC1);
	Mat cannyMatb   = Mat::zeros(srcMat.size(), CV_8UC1);
	Mat cannyMath   = Mat::zeros(srcMat.size(), CV_8UC1);
	int i, j;
	double scaleL, scaleH;
	int threshold = 40;

	for(i=0; i<srcMat.rows; i++){
		uchar *ptr1 = srcMat.ptr<uchar>(i);
		uchar *ptr2 = srcMatGaL.ptr<uchar>(i);
		uchar *ptr3 = srcMatGaH.ptr<uchar>(i);
		for(j=0; j<srcMat.cols; j++){
			scaleL = pow(((float)ptr1[j] / 255), 0.455);
			scaleH = pow(((float)ptr1[j] / 255), 2.2);

			ptr2[j] = (uchar)(scaleL * 255);
			ptr3[j] = (uchar)(scaleH * 255);
		}
	}
	Canny(srcMat, tmpMat, threshold, threshold, 3);
	Canny(srcMatGaL, srcMatGaL, threshold, threshold, 3);
	Canny(srcMatGaH, srcMatGaH, threshold, threshold, 3);

	cannyMatAll = tmpMat | srcMatGaL | srcMatGaH;
	cannyMatb = srcMatGaL - (srcMatGaL & tmpMat);
	cannyMath = srcMatGaH - (srcMatGaL & tmpMat);

	for(i=0; i<REGIONNUMBER; i++){
		regionSize[i] = 0;
		tmpPixelAll[i] = 0;
		tmpPixelB[i] = 0;
		tmpPixelH[i] = 0;
		myVb[i] = 0;	
		myVh[i] = 0;
		regionsize[i] = 0;	
	}
	for(i=0; i<srcMat.rows; i++){
		uchar *ptr1 = cannyMatAll.ptr<uchar>(i);
		uchar *ptr2 = cannyMatb.ptr<uchar>(i);
		uchar *ptr3 = cannyMath.ptr<uchar>(i);
		uchar *ptr4 = lightSegMat.ptr<uchar>(i);
		for(j=0; j<srcMat.cols; j++){
			int tmp;
			tmp = ptr4[j];
			if(tmp > 5){
				tmp = tmp - 2;
			}
			tmpPixelAll[tmp] = tmpPixelAll[tmp] + ptr1[j];
			tmpPixelB[tmp]   = tmpPixelB[tmp]   + ptr2[j];
			tmpPixelH[tmp]   = tmpPixelH[tmp]   + ptr3[j];
			regionsize[tmp] += 1;
		}
	}

	for(i=0; i<REGIONNUMBER; i++){
		myVb[i] = tmpPixelB[i] / (tmpPixelAll[i] + 0.00000001);
		myVh[i] = tmpPixelH[i] / (tmpPixelAll[i] + 0.00000001);
		regionSize[i] = (double)regionsize[i] / (srcMat.rows * srcMat.cols);
		printf("Vb[%d]:%lf, Vh[%d]:%lf, regionSize[%d]:%lf\n", i, myVb[i], i, myVh[i], i, regionSize[i]);
	}
}


int caculateDDij(Mat &src1, Mat &src2){
	int m, n;
	int tmpDij = 0, tmp = 0, tmp2 = 0;
	float val1, val2, val3;

	for(m=0; m<256; m++){
		tmp = 0;
		Mat tmpMat3 = Mat::zeros(1, 256, CV_32FC1);
		Mat tmpMat4 = Mat::zeros(1, 256, CV_32FC1);

		for(n=0; n<256; n++){
			val1 = src1.at<float>(0, n);
			if(m+n<256){
				tmpMat3.at<float>(0, m+n) = val1;
			}
		}
		for(n=0; n<256; n++){
			val1 = tmpMat3.at<float>(0, n);
			val2 = src2.at<float>(0, n);
			tmp += min((int)val1, (int)val2); 
		}
		if(tmp > tmp2){
			tmpDij = m;
			tmp2 = tmp;
		}
	}

	return tmpDij;
}

void caculateDij(Mat &regionMat, Mat &srcMat){
	int i, j;
	int tmp, tmpDij = 0;

	for(i=0; i<REGIONNUMBER; i++){
		for(j=0; j<2; j++){
			relaxDij[i][j] = 0;
		}
		histogramMat[i] = Mat::zeros(1, 256, CV_32FC1);
	}

	for(i=0; i<regionMat.rows; i++){
		uchar *ptr1 = regionMat.ptr<uchar>(i);
		uchar *ptr2 = srcMat.ptr<uchar>(i);
		for(j=0; j<regionMat.cols; j++){
			int tmp = ptr1[j];
			if(tmp > 5){
				tmp = tmp -2;
			}
			histogramMat[tmp].at<float>(0, ptr2[j]) += 1;
		}
	}

	for(i=0; i<7; i++){
		if(i != 6){
			tmpDij = 0;

			tmpDij = caculateDDij(histogramMat[i], histogramMat[i+1]);
			relaxDij[i][1] = tmpDij;
			relaxDij[i+1][0] = tmpDij;
		}
		printf("relaxDij[%d][0]:%d  ", i, relaxDij[i][0]);
		printf("relaxDij[%d][1]:%d  \n", i, relaxDij[i][1]);
	}
}

double caculateEi(int a1, int a2, int a3, int a4, int a5, int a6, int a7){
	double Pi = 0, Ei = 0;
	double arr[7][2];
	int i;

	arr[0][0] = (double)a1/10; arr[0][1] = 0.0;
	arr[1][0] = (double)a3/10; arr[1][1] = 0.1;
	arr[2][0] = (double)a3/10; arr[2][1] = 0.2;
	arr[3][0] = (double)a4/10; arr[3][1] = 0.3;
	arr[4][0] = (double)a5/10; arr[4][1] = 0.4;
	arr[5][0] = (double)a6/10; arr[5][1] = 0.7;
	arr[6][0] = (double)a7/10; arr[6][1] = 0.8;

	for(i=0; i<7; i++){
		if(i<5){
			Pi = Pi + myVb[i] * regionSize[i] * (1 / (1 + exp(-(arr[i][0] - arr[i][1]))));
		}else{
			Pi = Pi + myVh[i] * regionSize[i] * (1 / (1 + exp(-(arr[i][1] - arr[i][0]))));
		}
	}
	Ei = -log(Pi);

	return Ei;
}

double caculatePij(int a1, int a2, int a3, int a4, int a5, int a6, int a7){
	double Pij = 0, Eij = 0;
	double arr[7][2];
	int i, newDij;
	double E = 2.718, X = 3.14;

	arr[0][0] = (double)a1/10; arr[0][1] = 0.0;
	arr[1][0] = (double)a3/10; arr[1][1] = 0.1;
	arr[2][0] = (double)a3/10; arr[2][1] = 0.2;
	arr[3][0] = (double)a4/10; arr[3][1] = 0.3;
	arr[4][0] = (double)a5/10; arr[4][1] = 0.4;
	arr[5][0] = (double)a6/10; arr[5][1] = 0.7;
	arr[6][0] = (double)a7/10; arr[6][1] = 0.8;

	double t1 = sqrt(2 * X * 0.15 * 0.15);
	double t2 = 2 * 0.15 * 0.15;

	for(i=0; i<7; i++){
		if(i==0){
			newDij = abs((arr[0][0] - arr[0][1]) * 255 - relaxDij[0][1]);

			double t3 = ((double)newDij / 255 - (double)relaxDij[i][1] / 255) * ((double)newDij/255 - (double)relaxDij[i][1] / 255);
			Pij += regionSize[i+1] * pow(E, - t3 / t2) / t1;
		}else if(i==6){
			newDij = (arr[i][0] - arr[i][1]) * 255 + relaxDij[i][0];

			double t3 = ((double)newDij / 255 - (double)relaxDij[i][0] / 255) * ((double)newDij/255 - (double)relaxDij[i][0] / 255);
			Pij += regionSize[i-1] * pow(E, - t3 / t2) / t1;
		}else{
			newDij = abs((arr[i][0] - arr[i][1]) * 255 - relaxDij[i][1]);

			double t3 = ((double)newDij / 255 - (double)relaxDij[i][0] / 255) * ((double)newDij/255 - (double)relaxDij[i][0] / 255);
			Pij += regionSize[i + 1] *  pow(E, - t3 / t2) / t1;

			newDij = (arr[i][0] - arr[i][1]) * 255 + relaxDij[i][0];
			t3 = ((double)newDij / 255 - (double)relaxDij[i][0] / 255) * ((double)newDij/255 - (double)relaxDij[i][0] / 255);

			Pij += regionSize[i - 1] * pow(E, - t3 / t2) / t1;
			
		}
	}
	Eij = -log(Pij);

	return Eij;
}

double argMinCalculate(int a1, int a2, int a3, int a4, int a5, int a6, int a7){
	double sum = 0;
	double Ei = 0;
	double Pi = 0;
	double scale = 0.7;

	Ei = caculateEi(a1, a2, a3, a4, a5, a6, a7);
	Pi = caculatePij(a1, a2, a3, a4, a5, a6, a7);

	sum = Ei + scale * Pi;

	return sum;
}	

void calculateZ(void){
	int i=0, j=1, k=1, m=1, n=1, q=0, s=1;

	int i1=0, j1=1, k1=2, m1=3, n1=4, q1=5, s1=6;
	double tmp=0, sum=0;
	int number = 0;

	for(i=i1; i<6; i++){
		for(j=j1; j<6; j++){
			if(j<i)
				continue;
			for(k=k1; k<6; k++){
				if(k<j)
					continue;
				for(m=m1; m<6; m++){
					if(m<k)
						continue;
					for(n=n1; n<6; n++){
						if(n<m)
							continue;
						for(q=q1; q<11; q++){
							for(s=s1; s<11; s++){
								if(s<q)
									continue;
								tmp = argMinCalculate(i, j, k, m, n, q, s);
								if((tmp < sum) || (sum == 0)){
									sum = tmp;
									arrRegion[0] = i;	
									arrRegion[1] = j;	
									arrRegion[2] = k;	
									arrRegion[3] = m;	
									arrRegion[4] = n;	
									arrRegion[5] = q;	
									arrRegion[6] = s;	
								}
							}
						}
					}
				}
			}
		}
	}
	printf("arr:%d, %d, %d, %d, %d, %d, %d\n", 
			arrRegion[0], arrRegion[1], arrRegion[2], arrRegion[3], arrRegion[4], arrRegion[5], arrRegion[6]);
}

void calculateQs(void){
	double e_old = 0, e_new = 0, f_val = 0;
	int i, j;
	Scalar s1;

	e_old = 0.1 + 0.2 + 0.3 + 0.4;
	e_new = (double)arrRegion[0]/10 + (double)arrRegion[1]/10 + (double)arrRegion[2]/10 
			+ (double)arrRegion[3]/10 + (double)arrRegion[4]/10;

	for(i=0; i<5; i++){
		double tmp = (double)i / 10;
		f_val += 5 * tmp * exp(-14 * pow(tmp, 1.6));
	}

	myQs = (e_new - e_old) * (regionSize[0] + regionSize[1] + regionSize[2] + regionSize[3] + regionSize[4]) * f_val;
	printf("Qs:%lf, f_val:%lf, e_old:%lf, e_new:%lf\n", myQs, f_val, e_old, e_new);
}

void calculateQh(void){
	double e_old = 0, e_new = 0, f_val = 0;
	int i, j;
	Scalar s1;

	e_old = 0.7 + 0.8;
	e_new = (double)arrRegion[5]/10 + (double)arrRegion[6]/10;

	for(i=7; i<9; i++){
		double tmp = (double)i / 10;
		f_val += 5 * tmp * exp(-14 * pow(tmp, 1.6));
	}

	myQh = (e_new - e_old) * (regionSize[5] + regionSize[6]) * f_val;
	printf("Qh:%lf, f_val:%lf, e_old:%lf, e_new:%lf\n", myQh, f_val, e_old, e_new);
}

Mat S_curve(Mat &src){
	int i, j;
	double f_vals = 0, f_valh = 0;
	double tmpMin, tmpMax;
	double scale;

	for(i=0; i<src.rows; i++){
		uchar *ptr1 = src.ptr<uchar>(i);
		for(j=0; j<src.cols; j++){
			float tmpValue = (float)ptr1[j] / 255.0;

			f_vals = 5 * tmpValue * exp(-14 * pow(tmpValue, 1.6));
			f_valh = 5 * tmpValue * exp(-14 * pow(1 - tmpValue, 1.6));
			ptr1[j] = (uchar)((tmpValue + myQs * f_vals - myQh * f_valh)*255);
		}
	}

	return src;
}

Mat detailEnhancement(Mat &myGFilter, Mat &src){
	int i, j;
	float tmpValue;

	myGFilter = src - myGFilter;

	for(i=0; i<src.rows; i++){
		uchar *ptr1 = src.ptr<uchar>(i);
		uchar *ptr2 = myGFilter.ptr<uchar>(i);
		for(j=0; j<src.cols; j++){
			tmpValue = (float)ptr1[j] / 255.0;
			tmpValue = tmpValue + (2 * tmpValue * (1 - tmpValue)) * ((float)ptr2[j] / 255.0);
			ptr1[j] = max(0, min(255, (int)(tmpValue * 255.0)));
		}
	}

	return src;
}


Mat colorAdjust(Mat src, Mat out){
    src.convertTo(src, CV_32F);
    out.convertTo(out, CV_32F);

    vector<Mat> channels;
    split(src, channels);
    Mat weight = (out+1.0) / (channels[0]+1.0);

    channels[0] = out;
    merge(channels, src);
    src.convertTo(src, CV_8U);

	cvtColor(src, src, COLOR_YUV2BGR);
	cvtColor(src, src, COLOR_BGR2HSV);
    vector<Mat> hsv_channels;
    split(src, hsv_channels);

    hsv_channels[1].convertTo(hsv_channels[1], CV_32FC1);
    hsv_channels[1] = hsv_channels[1].mul(weight);
    hsv_channels[1].convertTo(hsv_channels[1], CV_8UC1);
    merge(hsv_channels, out);
	cvtColor(out, out, COLOR_HSV2BGR);
    
    return out;
}

int main(int argc, char* argv[]){
	int k;

	if(argc < 2){
		printf("Please input rgb pic and mono pic!\n");
		return -1;
	}
	Mat mySrc = imread(argv[1]);
	Mat mySrcGray = imread(argv[1], 0);


/*******灰度小图做区域分割************/
	int num_ccs;
	Mat tmpMat, segMat;
	resize(mySrcGray, tmpMat, Size(mySrcGray.cols/16, mySrcGray.rows/16));

	segMat = mySegment(tmpMat, num_ccs);
	printf("num_ccs:%d\n", num_ccs);

/*******相同亮度区域层合并************/
	Mat lightSegMat = lightSameSeg(segMat, tmpMat, num_ccs);

/******计算高亮/低亮区域细节比例权重***********/
	myGammaDetile(lightSegMat, tmpMat);

/********计算直方图偏移权重********************/
	caculateDij(lightSegMat, tmpMat);

/***********计算gamma权重**********************/
	calculateZ();
	calculateQs();
	calculateQh();
	
/***********颜色空间转换和通道分离*************/
	cvtColor(mySrc, mySrc, COLOR_BGR2YUV);
	vector<Mat> channels;
	split(mySrc, channels);

/***********对Y通道gamma调整*******************/
	Mat new_y = S_curve(channels[0]);

/*************细节增强*************************/
	Mat myGFilter;
	GaussianBlur(new_y, myGFilter, Size(5,5), 0, 0);
	new_y = detailEnhancement(myGFilter, new_y);

/*****************颜色调整*********************/  
    Mat out = colorAdjust(mySrc, new_y);

	imwrite(argv[2], out);

	return 0;
}
