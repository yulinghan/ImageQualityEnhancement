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

#include <highgui.h>
#include <iostream>
#include <fstream>
#include <time.h>

#include "utils.h"

using namespace std;

//-------------------------------------------------------

static const int NCOLOURS = 8;
static const CvScalar COLOURS [] = {cvScalar(255,0,0), cvScalar(0,255,0), 
                                    cvScalar(0,0,255), cvScalar(255,255,0),
                                    cvScalar(0,255,255), cvScalar(255,0,255),
                                    cvScalar(255,255,255), cvScalar(0,0,0)};

//-------------------------------------------------------

//显示错误信息并终止程序
void error(const char *msg) 
{
  cout << "\nError: " << msg;
  getchar();
  exit(0);
}

//-------------------------------------------------------

//显示所提供的图像并等待按键响应
void showImage(const IplImage *img)
{
  cvNamedWindow("Surf", CV_WINDOW_AUTOSIZE); 
  cvShowImage("Surf", img);  
  cvWaitKey(0);
}

//-------------------------------------------------------

//在指定窗口显示所提供的图像并等待按键响应
void showImage(char *title,const IplImage *img)
{
  cvNamedWindow(title, CV_WINDOW_AUTOSIZE); 
  cvShowImage(title, img);  
  cvWaitKey(0);
}

//-------------------------------------------------------

//转换成单通道32位浮点型的灰度图像
IplImage *getGray(const IplImage *img)
{
  //检查提供的图像是否为空
  if (!img) error("Unable to create grayscale image.  No image supplied");

  IplImage* gray8, * gray32;

  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );

  if( img->nChannels == 1 )
    gray8 = (IplImage *) cvClone( img );
  else {
    gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
    cvCvtColor( img, gray8, CV_BGR2GRAY );
  }

  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}

//-------------------------------------------------------

//画出所有特征点向量
void drawIpoints(IplImage *img, vector<Ipoint> &ipts, int tailSize)
{
  Ipoint *ipt;
  float s, o;
  int r1, c1, r2, c2, lap;

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    ipt = &ipts.at(i);
    s = (2.5f * ipt->scale);
    o = ipt->orientation;
    lap = ipt->laplacian;
    r1 = fRound(ipt->y);
    c1 = fRound(ipt->x);
    c2 = fRound(s * cos(o)) + c1;
    r2 = fRound(s * sin(o)) + r1;

    if (o) //绿线表示方向
      cvLine(img, cvPoint(c1, r1), cvPoint(c2, r2), cvScalar(0, 255, 0));
    else  //绿点（if using upright version）
      cvCircle(img, cvPoint(c1,r1), 1, cvScalar(0, 255, 0),-1);

    if (lap == 1)
    { //蓝色圆圈表示亮背景上的暗斑
      cvCircle(img, cvPoint(c1,r1), fRound(s), cvScalar(255, 0, 0),1);
    }
    else if (lap == 0)
    { //红色圆圈表示暗背景上的亮斑
      cvCircle(img, cvPoint(c1,r1), fRound(s), cvScalar(0, 0, 255),1);
    }
    else if (lap == 9)
    { //绿色圆圈表示暗背景上的亮斑（原英文注释有误：Red circles indicate light blobs on dark backgrounds）
      cvCircle(img, cvPoint(c1,r1), fRound(s), cvScalar(0, 255, 0),1);
    }

    //画出dx、dy的指向
    if (tailSize)
    {
      cvLine(img, cvPoint(c1,r1),
        cvPoint(int(c1+ipt->dx*tailSize), int(r1+ipt->dy*tailSize)),
        cvScalar(255,255,255), 1);
    }
  }
}

//-------------------------------------------------------

//在图像上画出单个特征
void drawIpoint(IplImage *img, Ipoint &ipt, int tailSize)
{
  float s, o;
  int r1, c1, r2, c2, lap;

  s = (2.5f * ipt.scale);
  o = ipt.orientation;
  lap = ipt.laplacian;
  r1 = fRound(ipt.y);
  c1 = fRound(ipt.x);

  //绿线表示方向
  if (o)
  {
    c2 = fRound(s * cos(o)) + c1;
    r2 = fRound(s * sin(o)) + r1;
    cvLine(img, cvPoint(c1, r1), cvPoint(c2, r2), cvScalar(0, 255, 0));
  }
  else  //绿点（if using upright version）
    cvCircle(img, cvPoint(c1,r1), 1, cvScalar(0, 255, 0),-1);

  if (lap >= 0)
  { //篮圈圈表示暗背景上的亮斑
    cvCircle(img, cvPoint(c1,r1), fRound(s), cvScalar(255, 0, 0),1);
  }
  else
  { //红圈圈表示暗背景上的亮斑
    cvCircle(img, cvPoint(c1,r1), fRound(s), cvScalar(0, 0, 255),1);
  }

  //画出dx、dy的指向
  if (tailSize)
  {
    cvLine(img, cvPoint(c1,r1),
      cvPoint(int(c1+ipt.dx*tailSize), int(r1+ipt.dy*tailSize)),
      cvScalar(255,255,255), 1);
  }
}

//-------------------------------------------------------

//在特征位置上画一个点
void drawPoint(IplImage *img, Ipoint &ipt)
{
  float s, o;
  int r1, c1;

  s = 3;
  o = ipt.orientation;
  r1 = fRound(ipt.y);
  c1 = fRound(ipt.x);

  cvCircle(img, cvPoint(c1,r1), fRound(s), COLOURS[ipt.clusterIndex%NCOLOURS], -1);
  cvCircle(img, cvPoint(c1,r1), fRound(s+1), COLOURS[(ipt.clusterIndex+1)%NCOLOURS], 2);

}

//-------------------------------------------------------

//在所有特征位置上画点
void drawPoints(IplImage *img, vector<Ipoint> &ipts)
{
  float s, o;
  int r1, c1;

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    s = 3;
    o = ipts[i].orientation;
    r1 = fRound(ipts[i].y);
    c1 = fRound(ipts[i].x);

    cvCircle(img, cvPoint(c1,r1), fRound(s), COLOURS[ipts[i].clusterIndex%NCOLOURS], -1);
    cvCircle(img, cvPoint(c1,r1), fRound(s+1), COLOURS[(ipts[i].clusterIndex+1)%NCOLOURS], 2);
  }
}

//-------------------------------------------------------

//用特征点向量画出描述符窗口
void drawWindows(IplImage *img, vector<Ipoint> &ipts)
{
  Ipoint *ipt;
  float s, o, cd, sd;
  int x, y;
  CvPoint2D32f src[4];

  for(unsigned int i = 0; i < ipts.size(); i++) 
  {
    ipt = &ipts.at(i);
    s = (10 * ipt->scale);
    o = ipt->orientation;
    y = fRound(ipt->y);
    x = fRound(ipt->x);
    cd = cos(o);
    sd = sin(o);

    src[0].x=sd*s+cd*s+x;   src[0].y=-cd*s+sd*s+y;
    src[1].x=sd*s+cd*-s+x;  src[1].y=-cd*s+sd*-s+y;
    src[2].x=sd*-s+cd*-s+x; src[2].y=-cd*-s+sd*-s+y;
    src[3].x=sd*-s+cd*s+x;  src[3].y=-cd*-s+sd*s+y;

    if (o) //画方向线
      cvLine(img, cvPoint(x, y), 
      cvPoint(fRound(s*cd + x), fRound(s*sd + y)), cvScalar(0, 255, 0),1);
    else  //绿点（if using upright version）
      cvCircle(img, cvPoint(x,y), 1, cvScalar(0, 255, 0),-1);


    //在特征点画个框框
    cvLine(img, cvPoint(fRound(src[0].x), fRound(src[0].y)), 
      cvPoint(fRound(src[1].x), fRound(src[1].y)), cvScalar(255, 0, 0),2);
    cvLine(img, cvPoint(fRound(src[1].x), fRound(src[1].y)), 
      cvPoint(fRound(src[2].x), fRound(src[2].y)), cvScalar(255, 0, 0),2);
    cvLine(img, cvPoint(fRound(src[2].x), fRound(src[2].y)), 
      cvPoint(fRound(src[3].x), fRound(src[3].y)), cvScalar(255, 0, 0),2);
    cvLine(img, cvPoint(fRound(src[3].x), fRound(src[3].y)), 
      cvPoint(fRound(src[0].x), fRound(src[0].y)), cvScalar(255, 0, 0),2);

  }
}

//-------------------------------------------------------

//在图像上画FPS图(至少需要2次调用)
void drawFPS(IplImage *img)
{
  static int counter = 0;
  static clock_t t;
  static float fps;
  char fps_text[20];
  CvFont font;
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1.0,1.0,0,2);

  //添加fps图（每10帧）
  if (counter > 10)
  {
    fps = (10.0f/(clock()-t) * CLOCKS_PER_SEC);
    t=clock(); 
    counter = 0;
  }

  //计数器自加
  ++counter;

  //转化成字符串
  sprintf(fps_text,"FPS: %.2f",fps);

  //在图像上显示帧率字符串
  cvPutText (img,fps_text,cvPoint(10,25), &font, cvScalar(255,255,0));
}

//-------------------------------------------------------

//保存SURF特征到文件
void saveSurf(char *filename, vector<Ipoint> &ipts)
{
  ofstream outfile(filename);

  //输出描述符的长度
  outfile << "64\n";
  outfile << ipts.size() << "\n";

  //每行的输出格式：scale  x  y  des
  for(unsigned int i=0; i < ipts.size(); i++) 
  {
    outfile << ipts.at(i).scale << "  ";
    outfile << ipts.at(i).x << " ";
    outfile << ipts.at(i).y << " ";
    outfile << ipts.at(i).orientation << " ";
    outfile << ipts.at(i).laplacian << " ";
    outfile << ipts.at(i).scale << " ";
    for(int j=0; j<64; j++)
      outfile << ipts.at(i).descriptor[j] << " ";

    outfile << "\n";
  }

  outfile.close();
}

//-------------------------------------------------------

//从文件中载入SURF特征
void loadSurf(char *filename, vector<Ipoint> &ipts)
{
  int descriptorLength, count;
  ifstream infile(filename);

  //首先清除ipts向量
  ipts.clear();

  //读取特征长度/特征点数目
  infile >> descriptorLength;
  infile >> count;

  //对于每个特征点
  for (int i = 0; i < count; i++) 
  {
    Ipoint ipt;

    //读取
    infile >> ipt.scale; 
    infile >> ipt.x;
    infile >> ipt.y;
    infile >> ipt.orientation;
    infile >> ipt.laplacian;
    infile >> ipt.scale;

    //读取描述符组件
    for (int j = 0; j < 64; j++)
      infile >> ipt.descriptor[j];

    ipts.push_back(ipt);

  }
}

//-------------------------------------------------------

//-------------------------------------------------------
