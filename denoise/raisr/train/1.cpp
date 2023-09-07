#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
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

char img_path[500][1000];
char trainDataPath[20] = "Training/";
Mat QMat[filterNum], VMat[filterNum], HMat[filterNum];
Mat filterMat[filterNum];
Mat filterPixel[filterNum];

int readFileList(char *basePath){
	DIR *dir;  
	struct dirent *ptr;  
	int img_num=0;

	if((dir=opendir(basePath)) == NULL){  
		perror("Open dir error...");  
		exit(1);  
	}  

	while((ptr=readdir(dir)) != NULL){
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir  
			continue;  
		else if(ptr->d_type == 8){    ///file  
			strcpy(img_path[img_num],basePath);  
			strcat(img_path[img_num++],ptr->d_name);  
		}else{
			continue;  
		}  
	}  
	closedir(dir);  
	return img_num;
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

void calGradAngStrengthCoherence(Mat &tiX, Mat &tiY, Mat &srcLR, Mat &srcHR){
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
	Mat xi = Mat::zeros(1, 1, CV_32FC1);

	for(i=0;i<window;i++){
		matB.at<float>(i,i) = gaussValue[i];
	}

	for(i=0;i<filterNum;i++){
		filterMat[i] = Mat::zeros(1, 1, CV_32FC1);
		filterPixel[i] = Mat::zeros(1, 1, CV_32FC1);
	}

	for(i=0;i<tiX.rows;i++){
		float *ptr7 = srcHR.ptr<float>(i);

		for(j=0;j<tiX.cols;j++){

			float *ptr3 = matA.ptr<float>(0);
			float *ptr4 = matA.ptr<float>(1);

			xi.at<float>(0,0) = ptr7[j];
			
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
				r1 = sqrt(eValuesMat.at<float>(0,0));
				r2 = sqrt(eValuesMat.at<float>(1,0));
			}else{
				eValueAddr = 1;
				r1 = sqrt(eValuesMat.at<float>(1,0));
				r2 = sqrt(eValuesMat.at<float>(0,0));
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

/********angle:angK, r1:Strength, uK:Coherence, pixelType, pk, xi*************/			
			curfilterIndex = getFilterIndex((int)angK, (int)r1, (int)uK, (int)pixelType);
//			printf("angK:%d, r1:%d, uK:%d, pixelType:%d, curfilterIndex:%d\n", (int)angK, (int)r1, (int)uK, (int)pixelType, curfilterIndex);

			if(filterMat[curfilterIndex].cols ==1){
				filterMat[curfilterIndex] = pK;
				filterPixel[curfilterIndex] = xi;
			}else{
				filterMat[curfilterIndex] = mergeRows(filterMat[curfilterIndex], pK);
				filterPixel[curfilterIndex] = mergeRows(filterPixel[curfilterIndex], xi);
			}
		}
	}
}

int main(int argc, char* argv[]){
	int i, j, k;

	if(argc<2){
		printf("Please input scale!\n");
		return -1;
	}
	
	int imgNum = readFileList(trainDataPath);
	float scale = atof(argv[1]);

/***********目前只能2倍训练***********/
	scale = 2.0;
	
	for(i=0;i<filterNum;i++){
		QMat[i] = Mat::zeros(window*window, window*window, CV_32FC1);
		VMat[i] = Mat::zeros(window*window, 1, CV_32FC1);
	}

	for(i=0;i<imgNum;i++){
		printf("i:%d, trainPic:%s\n", i, img_path[i]);

		Mat srcHR = imread(img_path[i], 0);

		/***************LR图像获取**************/
		srcHR.convertTo(srcHR, CV_32FC1, 1/255.0);

		Mat srcLR;

		resize(srcHR, srcLR, Size(srcHR.cols/scale, srcHR.rows/scale), 0, 0, INTER_CUBIC);
		resize(srcLR, srcLR, Size(srcHR.cols, srcHR.rows));

		/*********梯度计算*******************/
		Mat tiX  = Mat::zeros(srcHR.size(), CV_32FC1);
		Mat tiY  = Mat::zeros(Size(srcHR.rows, srcHR.cols), CV_32FC1);
		calculateFilter(tiX, tiY, srcLR);
		
		/*******计算梯度角度强度相关度*****/
		calGradAngStrengthCoherence(tiX, tiY, srcLR, srcHR);

		for(j=0;j<filterNum;j++){
			if(filterMat[j].cols != 1){
				QMat[j] += filterMat[j].t() * filterMat[j];
				VMat[j] += filterMat[j].t() * filterPixel[j];
			}
		}
	}

	for(i=0;i<filterNum;i++){
		solve(QMat[i], VMat[i], HMat[i]);
	}

	char str1[30] =  "../TrainData/HMat_";
	char str2[10];
	char str3[10] = ".xml";
	char strName[100];
	
	for(i=0;i<filterNum;i++){
		memset(strName, 0, sizeof(strName));
		sprintf(str2, "%d", i);
		strcat(strName, str1);
		strcat(strName, str2);
		strcat(strName, str3);

		FileStorage fs(strName, FileStorage::WRITE);
		fs << "vocabulary" << HMat[i];
		fs.release();
	}
	
	return 0;
}
