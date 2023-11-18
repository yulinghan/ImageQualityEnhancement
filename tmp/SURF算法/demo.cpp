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

/***********************************************************
*  translated by Mr.Hu(Graduate Student of SWJTU)		   *
*  His E-mail is eleftheria@163.com						   *
************************************************************/

#include "surflib.h"
#include "kmeans.h"
#include <ctime>
#include <iostream>

//-------------------------------------------------------
//为方便您能使用OpenSURF,对你能做一些简单的任务做如下说明。
//您只能使用其中一个函数来实现SURF特征提取!
//通过宏定义PROCEDURE的值来明确：
//  - 1 在提供的路径的静态图片上使用
//  - 2 从网络摄像头获取
//  - 3 在图片中找到匹配的目标
//  - 4 显示移动特征
//  - 5 显示静态图像之间的匹配
#define PROCEDURE 1

//-------------------------------------------------------
//  - 1 在提供的路径的静态图片上使用
int mainImage(void)
{
  //声明描述Ipoints的向量（Ipoints=Interest points，即：兴趣点/关键点/特征点）
  IpVec ipts;
  IplImage *img=cvLoadImage("imgs/sf.jpg");//载入图片（静态图）

  //检测图像中的特征点，其特征描述存放于向量ipts中
  clock_t start = clock();//计时开始
  surfDetDes(img, ipts, false, 5, 4, 2, 0.0004f); 
  clock_t end = clock();//停止计时

  std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;//打印出找到的特征点个数
  std::cout<< "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC  << " seconds" << std::endl;//打印算法用时

  // 画出检测到的特征点
  drawIpoints(img, ipts);
  
  // 显示结果
  showImage(img);

  return 0;
}

//-------------------------------------------------------
//  - 2 从网络摄像头获取
int mainVideo(void)
{
  //初始化摄像头捕捉设备
  CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
  if(!capture) error("No Capture");

  // 初始化视频写入
  //cv::VideoWriter vw("c:\\out.avi", CV_FOURCC('D','I','V','X'),10,cvSize(320,240),1);
  //vw << img;

  //创建一个窗口 
  cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE );

  //声明特征向量
  IpVec ipts;
  IplImage *img=NULL;

  //主循环
  while( 1 ) 
  {
    //捕捉一帧画面
    img = cvQueryFrame(capture);

    //提取surf特征点
    surfDetDes(img, ipts, false, 4, 4, 2, 0.004f);    

    //画出检测到的特征
    drawIpoints(img, ipts);

    //绘制帧图片
    drawFPS(img);

    //显示结果
    cvShowImage("OpenSURF", img);

    //按Esc键结束循环
    if( (cvWaitKey(10) & 255) == 27 ) break;
  }

  cvReleaseCapture( &capture );
  cvDestroyWindow( "OpenSURF" );
  return 0;
}


//-------------------------------------------------------

//  - 3 在图片中找到匹配的目标
int mainMatch(void)
{
  //初始化视频捕获设备
  CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
  if(!capture) error("No Capture");

  //声明特征点向量
  IpPairVec matches;
  IpVec ipts, ref_ipts;
  
  //object.jpg是我们希望在视频帧中寻找的参考目标
  IplImage *img = cvLoadImage("imgs/object.jpg"); //载入参考图像
  if (img == NULL) error("Need to load reference image in order to run matching procedure");
  CvPoint src_corners[4] = {{0,0}, {img->width,0}, {img->width, img->height}, {0, img->height}};
  CvPoint dst_corners[4];

  //提取参考对象的特征点向量
  surfDetDes(img, ref_ipts, false, 3, 4, 3, 0.004f);
  drawIpoints(img, ref_ipts);
  showImage(img);

  //创建窗口 
  cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE );

  //主循环捕获
  while( true ) 
  {
    //捕获一帧画面
    img = cvQueryFrame(capture);
     
    //提取帧画面中的特征点向量
    surfDetDes(img, ipts, false, 3, 4, 3, 0.004f);

    //进行向量匹配
    getMatches(ipts,ref_ipts,matches);

	//目标在视频帧图像中的位置
    if (translateCorners(matches, src_corners, dst_corners))
    {
	  //在目标上画框
      for(int i = 0; i < 4; i++ )
      {
        CvPoint r1 = dst_corners[i%4];
        CvPoint r2 = dst_corners[(i+1)%4];
        cvLine( img, cvPoint(r1.x, r1.y),
          cvPoint(r2.x, r2.y), cvScalar(255,255,255), 3 );
      }

      for (unsigned int i = 0; i < matches.size(); ++i)
        drawIpoint(img, matches[i].first);
    }

    //绘制帧图片
    drawFPS(img);

    //显示结果
    cvShowImage("OpenSURF", img);

    //按Esc键结束循环
    if( (cvWaitKey(10) & 255) == 27 ) break;
  }

  cvReleaseCapture( &capture );
  cvDestroyWindow( "OpenSURF" );
  return 0;
}


//-------------------------------------------------------

//  - 4 显示移动特征
int mainMotionPoints(void)
{
  //初始化视频捕获设备
  CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
  if(!capture) error("No Capture");

  //创建一个窗口 
  cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE );

  //声明Ipoints和其他东西
  IpVec ipts, old_ipts, motion;
  IpPairVec matches;
  IplImage *img;

  //
  while( 1 ) 
  {
	//抓取捕获源的帧
    img = cvQueryFrame(capture);

    //检测和描述一帧画面的Ipoints
    old_ipts = ipts;
    surfDetDes(img, ipts, true, 3, 4, 2, 0.0004f);

    //进行匹配
    getMatches(ipts,old_ipts,matches);
    for (unsigned int i = 0; i < matches.size(); ++i) 
    {
      const float & dx = matches[i].first.dx;
      const float & dy = matches[i].first.dy;
      float speed = sqrt(dx*dx+dy*dy);
      if (speed > 5 && speed < 30) 
        drawIpoint(img, matches[i].first, 3);
    }
        
    //显示结果
    cvShowImage("OpenSURF", img);

    //按Esc键结束循环
    if( (cvWaitKey(10) & 255) == 27 ) break;
  }

  //释放捕获设备
  cvReleaseCapture( &capture );
  cvDestroyWindow( "OpenSURF" );
  return 0;
}


//-------------------------------------------------------
//  - 5 显示静态图像之间的匹配
int mainStaticMatch()
{
  IplImage *img1, *img2;
  img1 = cvLoadImage("imgs/img1.jpg");
  img2 = cvLoadImage("imgs/img2.jpg");

  IpVec ipts1, ipts2;
  clock_t start = clock();//计时开始

  surfDetDes(img1,ipts1,false,4,4,2,0.0001f);
  surfDetDes(img2,ipts2,false,4,4,2,0.0001f);

  IpPairVec matches;
  getMatches(ipts1,ipts2,matches);
  clock_t end = clock();//停止计时
  for (unsigned int i = 0; i < matches.size(); ++i)
  {
    drawPoint(img1,matches[i].first);
    drawPoint(img2,matches[i].second);
  
    const int & w = img1->width;
    cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
    cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
  }

  std::cout<< "Matches: " << matches.size() <<std::endl;
  std::cout<< "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC  << " seconds" << std::endl;//打印算法用时

  cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
  cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
  cvShowImage("1", img1);
  cvShowImage("2",img2);
  cvWaitKey(0);

  return 0;
}

//-------------------------------------------------------

int mainKmeans(void)
{
  IplImage *img = cvLoadImage("imgs/img1.jpg");
  IpVec ipts;
  Kmeans km;
  
  //获取特征点
  surfDetDes(img,ipts,true,3,4,2,0.0006f);

  for (int repeat = 0; repeat < 10; ++repeat)
  {

    IplImage *img = cvLoadImage("imgs/img1.jpg");
    km.Run(&ipts, 5, true);
    drawPoints(img, km.clusters);

    for (unsigned int i = 0; i < ipts.size(); ++i)
    {
      cvLine(img, cvPoint(ipts[i].x,ipts[i].y), cvPoint(km.clusters[ipts[i].clusterIndex].x ,km.clusters[ipts[i].clusterIndex].y),cvScalar(255,255,255));
    }

    showImage(img);
  }

  return 0;
}

//-------------------------------------------------------

int main(void) 
{
  if (PROCEDURE == 1) return mainImage();
  if (PROCEDURE == 2) return mainVideo();
  if (PROCEDURE == 3) return mainMatch();
  if (PROCEDURE == 4) return mainMotionPoints();
  if (PROCEDURE == 5) return mainStaticMatch();
  if (PROCEDURE == 6) return mainKmeans();
}
