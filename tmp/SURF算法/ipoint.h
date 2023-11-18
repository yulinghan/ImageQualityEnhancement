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

#ifndef IPOINT_H
#define IPOINT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <math.h>


//-------------------------------------------------------

class Ipoint; //Ipoint类声明，主要定义关键点的相应属性
typedef std::vector<Ipoint> IpVec;
typedef std::vector<std::pair<Ipoint, Ipoint> > IpPairVec;

//-------------------------------------------------------

//Ipoint操作
void getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches);
int translateCorners(IpPairVec &matches, const cv::Point src_corners[4], cv::Point dst_corners[4]);

//-------------------------------------------------------

class Ipoint {

public:

  //析构函数
  ~Ipoint() {};

  //构造函数
  Ipoint() : orientation(0) {};

  //得到特征描述符之间的空间距离
  float operator-(const Ipoint &rhs)
  {
    float sum=0.f;
    for(int i=0; i < 64; ++i)//surf有64维特征，此处计算的是欧氏距离
      sum += (this->descriptor[i] - rhs.descriptor[i])*(this->descriptor[i] - rhs.descriptor[i]);
    return sqrt(sum);
  };

  //特征点坐标
  float x, y;

  //特征尺度
  float scale;

  //以x轴正方向，绕逆时针旋转的角度
  float orientation;

  //拉普拉斯算子
  int laplacian;

  //64维特征向量
  float descriptor[64];

  //! Placeholds for point motion (可以用于帧与帧之间的运动分析)
  float dx, dy;

  //索引
  int clusterIndex;
};

//-------------------------------------------------------


#endif
