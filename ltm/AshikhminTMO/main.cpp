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
#include "AshikhminTMO.hpp"

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
    float gamma = 1/3.0;
    float ashikhmin_smax = 3;

    Mat img = read_raw(argv[1]);
    imshow("src", img);

    AshikhminTMO *my_tonemap_mantiuk_test = new AshikhminTMO(gamma, ashikhmin_smax);
    Mat out = my_tonemap_mantiuk_test->Run(img);
    imshow("out", out);
    waitKey(0);

    imwrite(argv[2], out*255);
    return 0;
}
