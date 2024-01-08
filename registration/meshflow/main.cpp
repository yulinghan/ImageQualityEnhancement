#include <opencv2/opencv.hpp>
#include <iostream>
#include "SGridTracker.h"
#include "MeshFlow.h"
using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
    Mat src = imread(argv[1], 0);
    Mat ref = imread(argv[2], 0);
    
    resize(src, src, src.size()/4);
    resize(ref, ref, src.size());
    imwrite("src.jpg", src);
    imwrite("ref.jpg", ref);

	vector<cv::Mat> frames;
    frames.push_back(ref);
    frames.push_back(src);

	Mat dst = Mat::zeros(frames[0].size(), CV_8UC3);
	int width = dst.cols;
	int height = dst.rows;

	MeshFlow mf(width, height);
	Mesh *source, *destin;
	
	GridTracker gt;
	gt.trackerInit();
    gt.GetKeyPoints(frames[0]);
    
    gt.Update(frames[0], frames[1]);

    Mat show1 = gt.CornersShow(frames[0], gt.trackedFeas);
    Mat show2 = gt.CornersShow(frames[1], gt.preFeas);
    imshow("s1", show1);
    imshow("s2", show2);

    mf.ReInitialize();
    mf.SetFeature(gt.trackedFeas, gt.preFeas);
    mf.Execute();

    destin = mf.GetDestinMesh();
    source = mf.GetSourceMesh();

    dst.setTo(0);
    meshWarpRemap(frames[1], dst, *source, *destin);
    destin->drawMesh(dst); 
    cv::imwrite(argv[3], dst);

    imshow("111", dst);
    waitKey(0);

    return 0;
}
