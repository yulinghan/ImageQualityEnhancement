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
#include "ChiuTMO.hpp"

Mat read_raw(char* argv) {
    int height   = 384;//原始图像的高
	int width    = 512;//原始图像的宽
    int channels = 3  ;//原图像通道数

	FILE *fp = NULL; //定义指针
	fp = fopen(argv, "rb+");
	float *data = (float *)malloc(width*height*channels* sizeof(float)); //内存分配，new，malloc都行
	fread(data, sizeof(float), width*height*channels, fp); //在缓存中读取数据

	cv::Mat img;
	int bufLen = width*height*channels;  //定义长度
	img.create(height, width, CV_32FC3);//创建Mat
	memcpy(img.data, data, bufLen * sizeof(float)); //内存拷贝

    return img;
}

int main(int argc, char* argv[]) {
    Mat img = read_raw(argv[1]);
    resize(img, img, img.size()/2);
    imshow("src", img);

    ChiuTMO *my_chiu_test = new ChiuTMO();
    Mat out = my_chiu_test->Run(img);
    imshow("out", out);
    waitKey(0);

    //imwrite(argv[2], out);
    return 0;
}
