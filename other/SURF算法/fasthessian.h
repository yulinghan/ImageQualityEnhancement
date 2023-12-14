/*********************************************************** 
*  --- OpenSURF ---                                       *
*  This library is distributed under the GNU GPL. Please   *
*  use the contact form at http://www.chrisevansdev.com    *
*  for more information.                                   *
*                                                          *
*  C. Evans, Research Into Robust Visual Features,         *
*  MSc University of Bristol, 2008.                        *
*                                                          *
************************************************************/

#ifndef FASTHESSIAN_H
#define FASTHESSIAN_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ipoint.h"

#include <vector>

class ResponseLayer;
static const int OCTAVES = 5;//组数默认设置为5
static const int INTERVALS = 4;//层数默认设置为4
static const float THRES = 0.0004f;//斑点响应阈值默认设置为0.0004
static const int INIT_SAMPLE = 2;//采样间隔为2


class FastHessian {
  
  public:
   
	//构造函数（不带图像）
    FastHessian(std::vector<Ipoint> &ipts, 
                const int octaves = OCTAVES, 
                const int intervals = INTERVALS, 
                const int init_sample = INIT_SAMPLE, 
                const float thres = THRES);

	//构造函数（带图像）
    FastHessian(IplImage *img, 
                std::vector<Ipoint> &ipts, 
                const int octaves = OCTAVES, 
                const int intervals = INTERVALS, 
                const int init_sample = INIT_SAMPLE, 
                const float thres = THRES);

    //析构函数
    ~FastHessian();

	//保存参数
    void saveParameters(const int octaves, 
                        const int intervals,
                        const int init_sample, 
                        const float thres);

	//设置或重新设定积分图像来源
    void setIntImage(IplImage *img);

	//获取图像的特性并写入到特征向量中
    void getIpoints();
    
  private:

    //---------------- 私有函数 -----------------//

    //建立DoH响应表
    void buildResponseMap();

	//计算r层的DoH响应
    void buildResponseLayer(ResponseLayer *r);

    //在3x3x3空间中寻找极值点
    int isExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);    
    
	//插值函数 - 改编自Lowe的SIFT算法
    void interpolateExtremum(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
    void interpolateStep(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b,
                          double* xi, double* xr, double* xc );
    CvMat* deriv3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);
    CvMat* hessian3D(int r, int c, ResponseLayer *t, ResponseLayer *m, ResponseLayer *b);

    //---------------- 私有变量 -----------------//

	//积分图像指针及其属性
    IplImage *img;
    int i_width, i_height;

	//外部参考向量的特征（Reference to vector of features passed from outside）
    std::vector<Ipoint> &ipts;

	//hessian行列式的响应值
    std::vector<ResponseLayer *> responseMap;

    //组数
    int octaves;

    //每组层数
    int intervals;

    //! Initial sampling step for Ipoint detection
	//初始特征检测的抽样间隔
    int init_sample;

	//斑点响应阈值
    float thresh;
};


#endif
