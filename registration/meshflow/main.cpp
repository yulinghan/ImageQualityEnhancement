#include <opencv2/opencv.hpp>
#include <iostream>
#include "SGridTracker.h"
#include "MeshFlow.h"
using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1], 0);
    Mat ref = imread(argv[2], 0);
    
    resize(src, src, src.size()/8);
    resize(ref, ref, src.size());
    imwrite("src.jpg", src);
    imwrite("ref.jpg", ref);

	vector<cv::Mat> frames;
    frames.push_back(ref);
    frames.push_back(src);

	int width = src.cols;
	int height = src.rows;

	MeshFlow mf(width, height);
	Mesh *source, *destin;
	
	GridTracker gt;
    //一些初始化操作
	gt.trackerInit();

    //参考帧特征点提取
    gt.GetKeyPoints(frames[0]);

    //待配准图像特征点跟踪
    gt.Update(frames[0], frames[1]);

    //一些初始化操作
    mf.ReInitialize();

    //计算全局单应性矩阵
    //计算并存储每个块中配对特征点偏移
    mf.SetFeature(gt.trackedFeas, gt.preFeas);

    //根据网格内配对特征点信息，计算每个网格点配准偏移数据
    mf.Execute();

    //得到参考帧每个网格点原始坐标source
    source = mf.GetSourceMesh();

    //得到待配准图像每个网格点配准后数据
    destin = mf.GetDestinMesh();


    //每个网格根据它四个顶点数据,计算出对应的单应性矩阵，进行局域块图像配准
	Mat dst = Mat::zeros(frames[0].size(), CV_8UC3);
    meshWarpRemap(frames[1], dst, *source, *destin);
//    destin->drawMesh(dst); 
    cv::imwrite(argv[3], dst);


    return 0;
}
