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

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ipoint.h"

#include <vector>


//显示错误信息并终止程序
void error(const char *msg);

//显示所提供的图像并等待按键响应
void showImage(const IplImage *img);

//在指定窗口内显示所提供的图像并等待按键响应
void showImage(char *title,const IplImage *img);

//减图像转换为32位浮点型灰度图像
IplImage* getGray(const IplImage *img);

//在图像上画出单个特征
void drawIpoint(IplImage *img, Ipoint &ipt, int tailSize = 0);

//画出所有特征点向量
void drawIpoints(IplImage *img, std::vector<Ipoint> &ipts, int tailSize = 0);

//用特征点向量画出描述符窗口
void drawWindows(IplImage *img, std::vector<Ipoint> &ipts);

//在图像上画FPS图(至少需要2次调用)
void drawFPS(IplImage *img);

//在特征位置上画一个点
void drawPoint(IplImage *img, Ipoint &ipt);

//在所有特征位置上画点
void drawPoints(IplImage *img, std::vector<Ipoint> &ipts);

//保存SURF特征到文件
void saveSurf(char *filename, std::vector<Ipoint> &ipts);

//从文件中载入SURF特征
void loadSurf(char *filename, std::vector<Ipoint> &ipts);

//将浮点数转换成最接近的整数
inline int fRound(float flt)
{
  return (int) floor(flt+0.5f);
}

#endif
