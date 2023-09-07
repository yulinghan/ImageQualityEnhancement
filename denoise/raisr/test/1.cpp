#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv2/core/core.hpp>                                                                                                                                                                            
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <dirent.h>  
#include <unistd.h>

using namespace cv;
using namespace std;

#define window  9
#define filterNum   24*3*3*4

Mat FMat[filterNum];

void getTrainData(void){
	int i;
	
	for(i=0;i<filterNum;i++){
		char str1[30] =  "../TrainData/HMat_";
		char str2[10];
		char str3[10] = ".xml";
		char strName[100];

		memset(strName, 0, sizeof(strName));
		sprintf(str2, "%d", i);
		strcat(strName, str1);
		strcat(strName, str2);
		strcat(strName, str3);

		FileStorage fs(strName, FileStorage::READ);
		fs["vocabulary"] >> FMat[i];
	}
}


void calculateFilter(Mat &tiX, Mat &tiY, Mat &grayY){
	int i, j;
	float tmpTiValue[2], tmpDivValue[3];

	for(i=0;i<grayY.rows;i++){
		float *ptr1 = tiX.ptr<float>(i);
		float *ptr3 = grayY.ptr<float>(i);
		for(j=0;j<grayY.cols;j++){
			if(j<1){
				tmpTiValue[0] = 0;
				tmpTiValue[1] = ptr3[1];
			}else if(j>=(grayY.cols-1)){
				tmpTiValue[0] = ptr3[grayY.cols-2];
				tmpTiValue[1] = 0;	
			}else{
				tmpTiValue[0] = ptr3[j-1];
				tmpTiValue[1] = ptr3[j+1];
			}
			ptr1[j] = tmpTiValue[1] - tmpTiValue[0];
		}
	}

	Mat tmpMat = grayY.t();

	for(i=0;i<tmpMat.rows;i++){
		float *ptr1 = tiY.ptr<float>(i);
		float *ptr3 = tmpMat.ptr<float>(i);

		for(j=0;j<tmpMat.cols;j++){
			if(j<1){
				tmpTiValue[0] = 0;
				tmpTiValue[1] = ptr3[1];
			}else if(j==(tmpMat.cols-1)){
				tmpTiValue[0] = ptr3[tmpMat.cols-2];
				tmpTiValue[1] = 0;	
			}else{
				tmpTiValue[0] = ptr3[j-1];
				tmpTiValue[1] = ptr3[j+1];
			}
			ptr1[j] = tmpTiValue[1] - tmpTiValue[0];
		}
	}
	tiY = tiY.t();
}

Mat mergeRows(Mat A, Mat B){
	int totalRows = A.rows + B.rows;

	Mat mergedDescriptors(totalRows, A.cols, A.type());
	Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);

	return mergedDescriptors;
}

int getFilterIndex(int angK, int r1, int uK, int pixelType){
	int tmp1 = 24*3*3, tmp2 = 3*3, tmp3 = 3;

	int index = pixelType * tmp1;

	index += angK*tmp2;
	index += r1 * tmp3 + uK;

	return index;
}

Mat calGradAngStrengthCoherence(Mat &tiX, Mat &tiY, Mat &srcLR){
	int i, j, m, n;
	int tmpi, tmpj, eValueAddr, curfilterIndex=0;
	int winMove = window / 2;
	float tmpValue[2];

	Mat matA = Mat::zeros(2, window*window, CV_32FC1);
	Mat matB = Mat::zeros(window*window, window*window, CV_32FC1);
	float gaussValue[window] = {0.0001, 0.0044, 0.0540, 0.2420, 0.3989, 0.2420, 0.0540, 0.0044, 0.0001};
	Mat matC, eValuesMat, eVectorsMat;
	float angK, r1, r2, uK, pixelType;

	Mat pK = Mat::zeros(1, window*window, CV_32FC1);

	Mat out = Mat::zeros(tiX.size(), CV_32FC1);

	for(i=0;i<window;i++){
		matB.at<float>(i,i) = gaussValue[i];
	}

	for(i=0;i<tiX.rows;i++){
		printf("rows:%d, i:%d\n", tiX.rows, i);
		float *ptr7 = out.ptr<float>(i);
		float *ptr8 = srcLR.ptr<float>(i);
		for(j=0;j<tiX.cols;j++){
			float *ptr3 = matA.ptr<float>(0);
			float *ptr4 = matA.ptr<float>(1);

			if((i%2==0)&&(j%2==0)){
				pixelType = 0;
			}else if((i%2==0)&&(j%2!=0)){
				pixelType = 1;
			}else if((i%2!=0)&&(j%2==0)){
				pixelType = 2;
			}else{
				pixelType = 3;
			}

			for(m=-winMove;m<=winMove;m++){
				tmpi = i + m;
				tmpi = max(tmpi, 0);
				tmpi = min(tiX.rows-1, tmpi);

				float *ptr1 = tiX.ptr<float>(tmpi);
				float *ptr2 = tiY.ptr<float>(tmpi);
				float *ptr5 = srcLR.ptr<float>(tmpi);
				float *ptr6 = pK.ptr<float>(0);

				for(n=-winMove;n<=winMove;n++){
					tmpj = j + n;
					tmpj = max(tmpj, 0);
					tmpj = min(tiX.cols-1, tmpj);

					ptr3[(m+winMove)*window+n+winMove] = ptr1[tmpj];
					ptr4[(m+winMove)*window+n+winMove] = ptr2[tmpj];
					ptr6[(m+winMove)*window+n+winMove] = ptr5[tmpj];
				}
			}
			matC = matA * matB * matA.t();

			eigen(matC, eValuesMat, eVectorsMat);

			if(eValuesMat.at<float>(0,0) > eValuesMat.at<float>(1,0)){
				eValueAddr = 0;
				r1 = sqrt(abs(eValuesMat.at<float>(0,0)));
				r2 = sqrt(abs(eValuesMat.at<float>(1,0)));
			}else{
				eValueAddr = 1;
				r1 = sqrt(abs(eValuesMat.at<float>(1,0)));
				r2 = sqrt(abs(eValuesMat.at<float>(0,0)));
			}

			tmpValue[0] = eVectorsMat.at<float>(eValueAddr, 0);
			tmpValue[1] = eVectorsMat.at<float>(eValueAddr, 1);

			angK = atan(tmpValue[1]/(tmpValue[0]+ 0.000001)) * 180.0/3.1416 + 90;

			if(r1==0.0){
				uK = 0;
			}else{
				uK = (sqrt(r1) - sqrt(r2)) / (sqrt(r1) + sqrt(r2));
			}

			float tmpK, tmpr1, tmpuK;

			tmpK = angK;
			tmpr1 = r1;
			tmpuK = uK;

			angK = floor(angK / 7.5);  //angK范围[0,180],被分为24块，180/24 = 7.5;
			r1   = floor(r1 / 0.3334); //r1范围[0,1],被分为3块, 1/3 = 0.3333...;
			uK   = floor(uK / 0.3334); //uK范围[0,1],被分为3块, 1/3 = 0.3333...;

/********angle:angK, r1:Strength, uK:Coherence, pixelType, pk*************/			
			curfilterIndex = getFilterIndex((int)angK, (int)r1, (int)uK, (int)pixelType);
//			printf("curfilterIndex:%d, tmpK:%f, tmpr1:%f, r2:%f, tmpuK:%f\n", curfilterIndex, tmpK, tmpr1, r2, tmpuK);

			float tmpValue = 0;
			for(tmpi=0;tmpi<pK.cols;tmpi++){
				tmpValue += pK.at<float>(0, tmpi) * FMat[curfilterIndex].at<float>(tmpi,0);
			}

			if(tmpValue>1 && j>0){
				tmpValue = out.at<float>(i,j-1);
			}
			if(tmpValue<0.1 && j>0){
				tmpValue = out.at<float>(i,j-1);
			}

			ptr7[j] = tmpValue;
		}
	}

	return out;
}
Mat getCTPic(Mat &dst, Mat &srcLR){
	int i, j, m, n;
	int R=3, R1, tmpValue;
	int flag = 15;

	R1 = R/2;

	int arr[2][9];

	for(i=R;i<dst.rows-R;i++){
		uchar *ptr1 = dst.ptr(i);
		uchar *ptr2 = srcLR.ptr(i);
		for(j=R;j<dst.cols-R;j++){
			tmpValue = 0;
			for(m=-R1; m<=R1;m++){
				for(n=-R1; n<=R1;n++){
					if((m != 0) && (n != 0)){
						tmpValue += dst.at<float>(i+m, j+n);
					}
				}
			}
			tmpValue = tmpValue / 8;

			if((abs(ptr1[j]-tmpValue)>flag) && (abs(ptr1[j]-ptr2[j]) > flag)){
				ptr1[j] = ptr2[j];
			}
		}
	}

	return dst;
}

int main(int argc, char* argv[]){
	int i, j, k;

	if(argc<3){
		printf("Please input pic and scale!\n");
		return -1;
	}
	
/***********目前只能2倍训练***********/
	float scale = atof(argv[2]);
	scale = 2.0;
	
/*********装载滤波器字典*************/
	getTrainData();

/***************LR图像获取**************/
	Mat srcLR = imread(argv[1], 0);
	srcLR.convertTo(srcLR, CV_32FC1, 1/255.0);
	resize(srcLR, srcLR, Size(srcLR.cols*scale, srcLR.rows*scale), 0, 0, INTER_CUBIC);

/*********梯度计算*******************/
	Mat tiX  = Mat::zeros(srcLR.size(), CV_32FC1);
	Mat tiY  = Mat::zeros(Size(srcLR.rows, srcLR.cols), CV_32FC1);
	calculateFilter(tiX, tiY, srcLR);

    cout << "1111" << endl;

/*******计算梯度角度强度相关度*****/
	Mat dst = calGradAngStrengthCoherence(tiX, tiY, srcLR);
    cout << "222" << endl;

	dst.convertTo(dst, CV_8UC1, 255.0);
	srcLR.convertTo(srcLR, CV_8UC1, 255.0);

	imwrite("dst0.jpg", dst);
	dst = getCTPic(dst, srcLR);
    cout << "3333" << endl;

	imwrite("srcLR.jpg", srcLR);
	imwrite("dst.jpg", dst);

	return 0;
}
