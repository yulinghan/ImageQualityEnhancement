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

#ifndef SURFLIB_H
#define SURFLIB_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "integral.h"
#include "fasthessian.h"
#include "surf.h"
#include "ipoint.h"
#include "utils.h"


//函数功能：构建描述特征点的向量
inline void surfDetDes(IplImage *img,  /* 在此图上检测特征点 */
                       std::vector<Ipoint> &ipts, /* 存放特征点的特征向量 */
                       bool upright = false, /* 在旋转不变性的模式下运行 */
                       int octaves = OCTAVES, /* 金字塔的组数 */
                       int intervals = INTERVALS, /* 金字塔中每组的层数 */
                       int init_sample = INIT_SAMPLE, /* 初始抽样 */
                       float thres = THRES /* 斑点响应阈值 */)
{
  //创建积分图像
  IplImage *int_img = Integral(img);
  
  //创建快速Hessian对象
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);
 
  //提取特征点，并存放于ipts向量中
  fh.getIpoints();
  
  //创建surf特征描述符
  Surf des(int_img, ipts);

  //提取ipts中的描述符
  des.getDescriptors(upright);

  //释放积分图像
  cvReleaseImage(&int_img);
}


//函数功能：构建描述特征点的向量
inline void surfDet(IplImage *img,  /* 在此图上检测特征点 */
                    std::vector<Ipoint> &ipts, /* 存放特征点的特征向量 */
                    int octaves = OCTAVES, /* 金字塔的组数 */
                    int intervals = INTERVALS, /* 金字塔中每组的层数 */
                    int init_sample = INIT_SAMPLE, /* 初始抽样 */
                    float thres = THRES /* 斑点响应阈值 */)
{
  //创建积分图像
  IplImage *int_img = Integral(img);

  //创建快速Hessian对象
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

  //提取特征点，并存放于ipts向量中
  fh.getIpoints();

  //释放积分图像
  cvReleaseImage(&int_img);
}




//函数功能：特征点的向量描述
inline void surfDes(IplImage *img,  /* 在此图上检测特征点 */
                    std::vector<Ipoint> &ipts, /* 存放特征点的特征向量 */
                    bool upright = false) /* 在旋转不变性的模式下运行？ */
{ 
  //创建积分图像
  IplImage *int_img = Integral(img);

  //创建surf描述符对象
  Surf des(int_img, ipts);

  //提取ipts中的描述符
  des.getDescriptors(upright);
  
  //释放积分图像
  cvReleaseImage(&int_img);
}


#endif
