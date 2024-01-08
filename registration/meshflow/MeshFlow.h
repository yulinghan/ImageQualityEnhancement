#include "Mesh.h"

#ifndef __VECPT2F__
#define VecPt2f vector<cv::Point2f>
#endif __VECPT2F__

#ifndef __NODE__
#define __NODE__
struct node{
	VecPt2f features;
	vector<VecPt2f>MeshMotions;
	VecPt2f motions;
	vector<int>location_x;
	vector<int>location_y;
};
#endif

#ifndef __MESHFLOW__
#define __MESHFLOW__
class MeshFlow{

private:
	int m_height, m_width;
	int m_divide_x, m_divide_y;
	int m_quadWidth, m_quadHeight;
	int m_meshheight, m_meshwidth;

	cv::Mat m_source, m_target;
	cv::Mat m_globalHomography;
	Mesh* m_mesh;
	Mesh* m_warpedmesh;
	VecPt2f m_vertexMotion;
	node n;
	cv::Mat m_vertexMotionMatX;
	cv::Mat m_vertexMotionMatY;

private:
	void SpatialMedianFilter(int radius = 2);
	void DistributeMotion2MeshVertexes_MedianFilter();
	void WarpMeshbyMotion();
	cv::Point2f inline Trans(cv::Mat H, cv::Point2f &pt);

public:
	MeshFlow(int width, int height);
	~MeshFlow();
	void ReInitialize();
	void Execute(int radius=2);
	void SetFeature(vector<cv::Point2f> &spt, vector<cv::Point2f> &tpt);
	void GetMotions(cv::Mat &mapX, cv::Mat &mapY);
	cv::Mat GetVertexMotionX();
	cv::Mat GetVertexMotionY();
	void SetVertexMotion(const cv::Mat motionx, const cv::Mat motiony);
	int GetMeshHeight(){ return m_meshheight; }
	int GetMeshWidth(){ return m_meshwidth; }

	Mesh* GetDestinMesh(){ return m_warpedmesh; }
	Mesh* GetSourceMesh(){ return m_mesh; }
	void GetWarpedSource(cv::Mat &dst);
	void GetWarpedSource(cv::Mat &source, cv::Mat &dst);
};
#endif

void myQuickSort(vector<float> &arr, int left, int right);