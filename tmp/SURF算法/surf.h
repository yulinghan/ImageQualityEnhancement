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

#ifndef SURF_H
#define SURF_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ipoint.h"
#include "integral.h"

#include <vector>

class Surf {
  
  public:
    
	//标准构造函数（img是积分图像）
    Surf(IplImage *img, std::vector<Ipoint> &ipts);

	//在提供的向量中描述所有特征
    void getDescriptors(bool bUpright = false);
  
  private:
    
    //---------------- 私有函数 -----------------//

	//为当前Ipoint分配方向
    void getOrientation();
    
	//提取描述符【见论文：Agrawal ECCV 08】
    void getDescriptor(bool bUpright = false);

	//计算(x,y)处方差为sig的2维高斯值
    inline float gaussian(int x, int y, float sig);
    inline float gaussian(float x, float y, float sig);

	//计算x和y方向的Haar小波响应值
    inline float haarX(int row, int column, int size);
    inline float haarY(int row, int column, int size);

	//获取点(x,y)相对于正x轴方向的角度
    float getAngle(float X, float Y);


    //---------------- 私有变量 -----------------//

	//已检测到Ipoints的积分图像
    IplImage *img;

    //Ipoints向量
    IpVec &ipts;

	//当前Ipoint向量的索引值
    int index;
};


#endif
