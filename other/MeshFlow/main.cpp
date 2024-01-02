#include <opencv2/opencv.hpp>
#include <iostream>
#include "SGridTracker.h"
#include "MeshFlow.h"
using namespace std;

int main(){

	vector<cv::Mat> frames;
	char name[1024];
	for (int i = 1; i < 9; i++){
		sprintf(name, "../../data/registration/input/mult/%d.jpg",i);
        cv::Mat cur_mat = cv::imread(name,0);
		frames.push_back(cur_mat);
        resize(frames[i-1], frames[i-1], frames[i-1].size()/4);
	}
	cv::Mat dst = cv::Mat::zeros(frames[0].size(), CV_8UC3);
	int width = dst.cols;
	int height = dst.rows;

	MeshFlow mf(width, height);
	Mesh *source, *destin;
	
	GridTracker gt;
	gt.trackerInit(frames[0]);
	
	double t = (double)cv::getTickCount();
	for (int i = 1; i < frames.size(); i++){
		gt.Update(frames[0], frames[i]);
		
		mf.ReInitialize();
		mf.SetFeature(gt.trackedFeas, gt.preFeas);
		mf.Execute();
		destin = mf.GetDestinMesh();
		source = mf.GetSourceMesh();
		
		dst.setTo(0);
		sprintf(name, "../../data/registration/out/%d.jpg",i);
		meshWarpRemap(frames[i], dst, *source, *destin);
		destin->drawMesh(dst); 
		cv::imwrite(name, dst);
	}
	t = ((double)cv::getTickCount() - t) / (cv::getTickFrequency() * 1000);
	cout << "my code time: " << t << "ms" << endl;

	GridTracker gt1;
	gt1.trackerInit(frames[0]);
	gt1.Update(frames[0], frames[5]);
	vector<int> mask;
	cv::Mat H = cv::findHomography(gt1.trackedFeas, gt1.preFeas, mask, cv::RANSAC,4.0);
	for (int i = 0; i < mask.size(); i++){
		if (mask[i]){
			cv::circle(frames[0], gt1.preFeas[i], 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(frames[5], gt1.trackedFeas[i], 2, cv::Scalar(0, 255, 0), -1);
		}
	}

	MeshFlow mf1(width, height);
	mf1.SetFeature(gt.preFeas, gt.trackedFeas);
	mf1.Execute();
	meshWarpRemap(frames[0], dst, *mf1.GetSourceMesh(), *mf1.GetDestinMesh());

	cv::imwrite("../../data/registration/out/0.jpg",dst);
	cv::imwrite("../../data/registration/out/6.jpg",frames[5]);

    return 0;
}
