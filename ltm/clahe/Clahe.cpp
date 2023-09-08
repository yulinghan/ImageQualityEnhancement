#include "Clahe.hpp"

MyClaheTest::MyClaheTest() {
}

MyClaheTest::~MyClaheTest() {
}

vector<vector<float>> MyClaheTest::GetAdjustParam(Mat src, int width_block, int height_block, int step, float scale) {
	vector<vector<float>> hist_num;
	vector<vector<float>> hist_arr;

	for(int i=0; i<step*step; i++) {
		vector<float> cur_tmp;
		vector<float> cur_c2;
		for(int j=0; j<256; j++) {
			float cur_i = 0.0;
			cur_tmp.push_back(cur_i);
			float c_value = 0.0;
			cur_c2.push_back(c_value);

		}
		hist_num.push_back(cur_tmp);
		hist_arr.push_back(cur_c2);
	}

	int total = width_block * height_block; 
	for(int i=0;i<step;i++) {
		for(int j=0;j<step;j++) {
			int start_x = i*width_block;
			int end_x = start_x + width_block;
			int start_y = j*height_block;
			int end_y = start_y + height_block;
			int num = i+step*j;  

			for(int ii = start_x ; ii < end_x ; ii++) {  
				for(int jj = start_y ; jj < end_y ; jj++) {
					int index =src.at<uchar>(jj,ii);
					hist_num[num][index]++;  
				}  
			}

			int average = width_block * height_block / 255;  
			int LIMIT = scale * average;  
			int steal = 0;  
			for(int k = 0 ; k < 256 ; k++) {  
				if(hist_num[num][k] > LIMIT){  
					steal += hist_num[num][k] - LIMIT;  
					hist_num[num][k] = LIMIT;  
				}  
			}  
			int bonus = steal/256;  
			for(int k = 0 ; k < 256 ; k++) {  
				hist_num[num][k] += bonus;  
			}  

			for(int k = 0 ; k < 256 ; k++) {
				if( k == 0) { 
					hist_arr[num][k] = 1.0f * hist_num[num][k] / total;  
				} else {
					hist_arr[num][k] = hist_arr[num][k-1] + 1.0f * hist_num[num][k] / total;  
				}
			}  
		}
	}
	return hist_arr;
}

Mat MyClaheTest::GetAdjustMat(Mat src, vector<vector<float>> hist_arr, int width_block, int height_block, int step) {
	int width = src.cols;
	int height= src.rows;

	Mat out = Mat::zeros(src.size(), src.type());

	for(int  i = 0 ; i < width; i++) {  
		for(int j = 0 ; j < height; j++) {  
			if(i <= width_block/2 && j <= height_block/2) {  
				int num = 0;  
				out.at<uchar>(j,i) = (int)(hist_arr[num][src.at<uchar>(j,i)] * 255);  
			} else if(i <= width_block/2 && j >= ((step-1)*height_block + height_block/2)){  
				int num = step*(step-1);  
				out.at<uchar>(j,i) = (int)(hist_arr[num][src.at<uchar>(j,i)] * 255);  
			} else if(i >= ((step-1)*width_block+width_block/2) && j <= height_block/2){  
				int num = step-1;  
				out.at<uchar>(j,i) = (int)(hist_arr[num][src.at<uchar>(j,i)] * 255);  
			} else if(i >= ((step-1)*width_block+width_block/2) && j >= ((step-1)*height_block + height_block/2)){  
				int num = step*step-1;  
				out.at<uchar>(j,i) = (int)(hist_arr[num][src.at<uchar>(j,i)] * 255);  
			} else if( i <= width_block/2 ) {  
				int num_i = 0;  
				int num_j = (j - height_block/2)/height_block;  
				int num1 = num_j*step + num_i;  
				int num2 = num1 + step;  
				float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);  
				float q = 1-p;  
				out.at<uchar>(j,i) = (int)((q*hist_arr[num1][src.at<uchar>(j,i)]+ p*hist_arr[num2][src.at<uchar>(j,i)])* 255);  
			} else if( i >= ((step-1)*width_block+width_block/2)){  
				int num_i = step-1;  
				int num_j = (j - height_block/2)/height_block;  
				int num1 = num_j*step + num_i;  
				int num2 = num1 + step;  
				float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);  
				float q = 1-p;  
				out.at<uchar>(j,i) = (int)((q*hist_arr[num1][src.at<uchar>(j,i)]+ p*hist_arr[num2][src.at<uchar>(j,i)])* 255);  
			} else if( j <= height_block/2 ){  
				int num_i = (i - width_block/2)/width_block;  
				int num_j = 0;  
				int num1 = num_j*step + num_i;  
				int num2 = num1 + 1;  
				float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);  
				float q = 1-p;  
				out.at<uchar>(j,i) = (int)((q*hist_arr[num1][src.at<uchar>(j,i)]+ p*hist_arr[num2][src.at<uchar>(j,i)])* 255);  
			} else if( j >= ((step-1)*height_block + height_block/2) ){  
				int num_i = (i - width_block/2)/width_block;  
				int num_j = step-1;  
				int num1 = num_j*step + num_i;  
				int num2 = num1 + 1;  
				float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);  
				float q = 1-p;  
				out.at<uchar>(j,i) = (int)((q*hist_arr[num1][src.at<uchar>(j,i)]+ p*hist_arr[num2][src.at<uchar>(j,i)])* 255);
			} else{
				int num_i = (i - width_block/2)/width_block;  
				int num_j = (j - height_block/2)/height_block;  
				int num1 = num_j*step + num_i;  
				int num2 = num1 + 1;  
				int num3 = num1 + step;  
				int num4 = num2 + step;  
				float u = (i - (num_i*width_block+width_block/2))/(1.0f*width_block);  
				float v = (j - (num_j*height_block+height_block/2))/(1.0f*height_block);  
				out.at<uchar>(j,i) = (int)((u*v*hist_arr[num4][src.at<uchar>(j,i)] +   
							(1-v)*(1-u)*hist_arr[num1][src.at<uchar>(j,i)] +  
							u*(1-v)*hist_arr[num2][src.at<uchar>(j,i)] +  
							v*(1-u)*hist_arr[num3][src.at<uchar>(j,i)]) * 255); 
			}  
		}  
	}  
	return out;
}

Mat MyClaheTest::Run(Mat src, int step, float scale) {
	int width = src.cols;
	int height= src.rows;
	int width_block  = width/step;
	int height_block = height/step;

	vector<vector<float>> hist_arr = GetAdjustParam(src, width_block, height_block, step, scale);
	Mat out = GetAdjustMat(src, hist_arr, width_block, height_block, step);

	return out;
}
