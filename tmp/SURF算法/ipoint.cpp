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
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "ipoint.h"

//向量匹配
void getMatches(IpVec &ipts1, IpVec &ipts2, IpPairVec &matches)
{
  float dist, d1, d2;
  Ipoint *match;

  matches.clear();

  for(unsigned int i = 0; i < ipts1.size(); i++) 
  {
    d1 = d2 = FLT_MAX;

    for(unsigned int j = 0; j < ipts2.size(); j++) 
    {
      dist = ipts1[i] - ipts2[j];  

      if(dist<d1) //如果此特征匹配比当前最佳匹配还好
      {
        d2 = d1;
        d1 = dist;
        match = &ipts2[j];
      }
      else if(dist<d2) //如果此特征匹配比当前次佳匹配还好
      {
        d2 = dist;
      }
    }

	//如果d1:d2<0.65则认为是一匹配对
    if(d1/d2 < 0.65) 
    { 
	  //存储位置的变化
      ipts1[i].dx = match->x - ipts1[i].x; 
      ipts1[i].dy = match->y - ipts1[i].y;
      matches.push_back(std::make_pair(ipts1[i], *match));
    }
  }
}

//
//单应性（可理解为：透视变换）使用的是CV_RANSAC函数（OpenCV 1.1），在大多数linux发行版中不会编译
//
//-------------------------------------------------------

//求解匹配点之间的单应性矩阵，并将src_corners变换到dst_corners
int translateCorners(IpPairVec &matches, const cv::Point src_corners[4], cv::Point dst_corners[4])
{
#ifndef LINUX
  double h[9];//保存单应性矩阵的数组h
  CvMat _h = cvMat(3, 3, CV_64F, h);//将h转换成矩阵
  std::vector<CvPoint2D32f> pt1, pt2;
  CvMat _pt1, _pt2;
  
  int n = (int)matches.size();
  if( n < 4 ) return 0;

  //设置向量的大小
  pt1.resize(n);
  pt2.resize(n);

  //向量的复制
  for(int i = 0; i < n; i++ )
  {
    pt1[i] = cvPoint2D32f(matches[i].second.x, matches[i].second.y);
    pt2[i] = cvPoint2D32f(matches[i].first.x, matches[i].first.y);
  }
  _pt1 = cvMat(1, n, CV_32FC2, &pt1[0] );
  _pt2 = cvMat(1, n, CV_32FC2, &pt2[0] );

  //求解两个点集之间的单应性变换矩阵h
  if(!cvFindHomography(&_pt1, &_pt2, &_h, cv::RANSAC, 5))  //此行要求opencv 1.1
    return 0;

  //根据上面求得的单应性变换矩阵h，将src_corners变换到dst_corners
  for(int i = 0; i < 4; i++ )
  {
    double x = src_corners[i].x, y = src_corners[i].y;
    double Z = 1./(h[6]*x + h[7]*y + h[8]);
    double X = (h[0]*x + h[1]*y + h[2])*Z;
    double Y = (h[3]*x + h[4]*y + h[5])*Z;
    dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
  }
#endif
  return 1;
}


