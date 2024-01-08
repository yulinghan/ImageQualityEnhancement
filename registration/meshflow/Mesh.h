#pragma once

#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
using namespace std;

#ifndef __QUAD__
#define __QUAD__

class Quad
{
public:
	cv::Point2f V00;
	cv::Point2f V01;
	cv::Point2f V10;
	cv::Point2f V11;

	Quad();
	Quad(const Quad &inQuad);
	Quad(const cv::Point2f &inV00, const cv::Point2f &inV01, const cv::Point2f &inV10, const cv::Point2f &inV11);
	~Quad();

	void operator=(const Quad &inQuad);

	double getMinX() const;
	double getMaxX() const;
	double getMinY() const;
	double getMaxY() const;

	bool isPointIn(const cv::Point2f &pt) const;
	bool getBilinearCoordinates(const cv::Point2f &pt, vector<double> &coefficient) const;
	bool getBilinearCoordinates(const cv::Point2f &pt, double* &coefficient) const;
	inline void printQuad(){
		printf("V00 = %f %f\n", V00);
		printf("V01 = %f %f\n", V01);
		printf("V10 = %f %f\n", V10);
		printf("V11 = %f %f\n", V11);
	}

	cv::Point2f getPointByBilinearCoordinates(const vector<double> &coefficient) const;
};
#endif

#ifndef __MESH__
#define __MESH__

class Mesh
{
public:
	int imgRows;
	int imgCols;

	int width;
	int height;

	int quadWidth;
	int quadHeight;

	Mesh();
	Mesh(const Mesh &inMesh);
	Mesh(int rows, int cols);
	Mesh(int rows, int cols, double quadWidth, double quadHeight);
	~Mesh();

	void operator=(const Mesh &inMesh);
	double differentFrom(const Mesh &inMesh) const;

	cv::Point2f getVertex(int i, int j) const;
	Quad getQuad(int i, int j) const;
	void setVertex(int i, int j, const cv::Point2f &pos);

	void initialize(int w, int h);
	void buildMesh(double quadWidth, double quadHeight);

	void drawMesh(cv::Mat &targetImg);

	bool selfCheck();

	void smoothMesh();

	void HomographyTransformation(const cv::Mat &H);

	void UpdateMeshByMatrix(vector<cv::Mat>HList);

	void UpdateMeshByFlow(cv::Mat motionx, cv::Mat motiony);

private:
	cv::Mat xMat;
	cv::Mat yMat;
};
#endif

bool isPointInTriangular(const cv::Point2f &pt, const cv::Point2f &V0, const cv::Point2f &V1, const cv::Point2f &V2);
void CalculateMeshHomoLR(Mesh *srcmesh, Mesh *dstmesh, vector<cv::Mat>&meshHomo, cv::Mat Prewarp = cv::Mat::eye(3, 3, CV_64FC1));
void CalculateMeshHomoLR(Mesh srcmesh, Mesh dstmesh, vector<cv::Mat>&meshHomo, cv::Mat Prewarp = cv::Mat::eye(3, 3, CV_64FC1));
void meshWarp(const cv::Mat src, cv::Mat dst, const Mesh &m1, const Mesh &m2);
void quadWarp(const cv::Mat src, cv::Mat dst, const Quad &q1, const Quad &q2);
void meshWarp_multicore(const cv::Mat src, cv::Mat dst, const Mesh &m1, const Mesh &m2);
void meshWarpRemap(cv::Mat &src, cv::Mat &dst, Mesh &m1, Mesh &m2);
