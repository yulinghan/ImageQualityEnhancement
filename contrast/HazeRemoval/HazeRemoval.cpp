#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void makeDepth32f(Mat& source, Mat& output)
{
	if ((source.depth() != CV_32F) > FLT_EPSILON)
	source.convertTo(output, CV_32F);
	else
		output = source;
}


void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
	//CV_Assert(radius >= 2 && epsilon > 0);
	CV_Assert(source.data != NULL && source.channels() == 1);//可改变输入图像类型（通道）
	CV_Assert(guided_image.channels() == 1);                 //导向图一般为单通道
	CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

	Mat guided;
	if (guided_image.data == source.data) 
		guided_image.copyTo(guided);
	else
		guided = guided_image;
	
	Mat source_32f, guided_32f;
	makeDepth32f(source, source_32f);//将输入扩展为32位浮点型，以便以后做乘法
	makeDepth32f(guided, guided_32f);

	Mat mat_Ip, mat_I2;   //计算I*p和I*I
	multiply(guided_32f, source_32f, mat_Ip);
	multiply(guided_32f, guided_32f, mat_I2);
	
	Mat mean_p, mean_I, mean_Ip, mean_I2;   //计算各种均值
	Size win_size(2 * radius + 1, 2 * radius + 1);
	boxFilter(source_32f, mean_p, CV_32F, win_size);
	boxFilter(guided_32f, mean_I, CV_32F, win_size);
	boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
	boxFilter(mat_I2, mean_I2, CV_32F, win_size);
	
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);//计算Ip的协方差和I的方差
	Mat var_I = mean_I2 - mean_I.mul(mean_I);
	var_I += epsilon;
	
	Mat a, b;   //求a和b
	divide(cov_Ip, var_I, a);
	b = mean_p - a.mul(mean_I);
	
	Mat mean_a, mean_b;  //对包含像素i的所有a、b做平均
	boxFilter(a, mean_a, CV_32F, win_size);
	boxFilter(b, mean_b, CV_32F, win_size);
	
	output = mean_a.mul(guided_32f) + mean_b;//计算输出 (depth == CV_32F)
}


void HazeRemoval(Mat& image, Mat& imageRGB) {
	CV_Assert(!image.empty() && image.channels() == 3);
	Mat fImage;
	image.convertTo(fImage, CV_32FC3, 1.0 / 255, 0);//图片归一化
	
	Mat fImageBorder;
	int hPatch = 15,vPatch = 15;//设定最小滤波patch的大小,且均为奇数
	copyMakeBorder(fImage, fImageBorder, vPatch / 2, vPatch / 2, hPatch / 2, hPatch / 2, BORDER_REPLICATE);//给归一化的图片添加边界
	vector<Mat> fImageBorderVector(3);
	split(fImageBorder, fImageBorderVector);//分离通道
	
	Mat darkChannel(image.rows, image.cols, CV_32FC1);//创建darkChannel
	double minTemp, minPixel;
	for (unsigned int r = 0; r < darkChannel.rows; r++) {
		for (unsigned int c = 0; c < darkChannel.cols; c++) {
			minPixel = 1.0;
			for (vector<Mat>::iterator it = fImageBorderVector.begin(); it != fImageBorderVector.end(); it++) {
				Mat roi(*it, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			darkChannel.at<float>(r, c) = float(minPixel);
		}
	}
	
	//求出A(global atmospheric light),计算出darkChannel中前top个亮的值,论文中取值为0.1%
	float top = 0.001;
	float numberTop = top * darkChannel.rows*darkChannel.cols;
	Mat darkChannelVector;
	darkChannelVector = darkChannel.reshape(1, 1);//reshape的一个参数表示通道数，第二个表示矩阵行数
	Mat_<int> darkChannelVectorIndex;
	sortIdx(darkChannelVector, darkChannelVectorIndex, SORT_EVERY_ROW + SORT_DESCENDING);//降序，返回像素索引

	int count = 0, temp = 0;
	unsigned int x, y; //映射回暗通道图的像素位置
	Mat mask(darkChannel.rows, darkChannel.cols, CV_8UC1);//制作掩码,注意mask的类型必须是CV_8UC1
	for (unsigned int r = 0; r < darkChannelVectorIndex.rows; r++) {
		for (unsigned int c = 0; c < darkChannelVectorIndex.cols; c++) {
			temp = darkChannelVectorIndex.at<int>(r, c);
			x = temp / darkChannel.cols;
			y = temp % darkChannel.cols;

			if (count < numberTop) {
				mask.at<uchar>(x, y) = 1;
				count++;
			} else {
				mask.at<uchar>(x, y) = 0;
            }
		}
	}

	vector<double> A(3);                //分别存取B,G,R通道的最大A值
	vector<Mat> fImageBorderVectorA(3);//在求第三步的t(x)时，会用到以下的矩阵，这里可以提前求出
	vector<double>::iterator itA = A.begin();
	vector<Mat>::iterator it = fImageBorderVector.begin();
	vector<Mat>::iterator itAA = fImageBorderVectorA.begin();
	for (; it != fImageBorderVector.end() && itA != A.end() && itAA != fImageBorderVectorA.end(); it++, itA++, itAA++)
	{
		Mat roi(*it, Rect(0, 0, darkChannel.cols, darkChannel.rows));
		minMaxLoc(roi, 0, &(*itA), 0, 0, mask);
		(*itAA) = (*it) / (*itA); //注意：这个地方有除号，但是没有判断是否等于0,*itA等于零的可能性很小
	}

	//求出t(x)
	Mat darkChannelA(darkChannel.rows, darkChannel.cols, CV_32FC1);
	float omega = 0.95;      //论文中取值为0.95
	for (unsigned int r = 0; r < darkChannel.rows; r++) {
		for (unsigned int c = 0; c < darkChannel.cols; c++) {
			minPixel = 1.0;
			for (itAA = fImageBorderVectorA.begin(); itAA != fImageBorderVectorA.end(); itAA++) {
				Mat roi(*itAA, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			darkChannelA.at<float>(r, c) = float(minPixel);
		}
	}
	Mat tx1 = 1.0 - omega * darkChannelA;
	Mat tx(darkChannel.rows, darkChannel.cols, CV_32FC1);

	guidedFilter(tx1, tx1,tx, 8, 500);

	float t0 = 0.1;//论文中取0.1
	for (size_t r = 0; r < imageRGB.rows; r++) {
		for (size_t c = 0; c < imageRGB.cols; c++) {
			imageRGB.at<Vec3f>(r, c) = Vec3f((fImage.at<Vec3f>(r, c)[0] - A[0]) / max(tx.at<float>(r, c), t0) + A[0], (fImage.at<Vec3f>(r, c)[1] - A[1]) / max(tx.at<float>(r, c), t0) + A[1], (fImage.at<Vec3f>(r, c)[2] - A[2]) / max(tx.at<float>(r, c), t0) + A[2]);
		}
	}
}

int main(int argc, char* argv[]){
	Mat image = imread(argv[1]);

	Mat testimage(image.size(), CV_32FC3);
	HazeRemoval(image,testimage);

    testimage = testimage * 255;
    testimage.convertTo(testimage, CV_8UC1);
    imwrite(argv[2], testimage);

	return 0;
}
