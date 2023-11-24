
/**
 * @file Ipoint.h
 * @brief Class that defines a point of interest
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 * @update: 2013-03-28 by Yuhua Zou
 */

#ifndef _IPOINT_H_
#define _IPOINT_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include <vector>
#include <algorithm>
#include <math.h>
#include "opencv2/core/core.hpp"

// Ipoint Class Declaration
class Ipoint
{

public:

    // 特征点的浮点坐标和整数坐标（Coordinates of the detected interest point）
    float xf,yf;    // Float coordinates
    int x,y;        // Integer coordinates

    // 特征点的尺度级别，σ为单位（Detected scale of the interest point (sigma units)）
    float scale;

    // 图像尺度参数的整数值（Size of the image derivatives (pixel units)）
    int sigma_size;

    // 特征检测响应值（Feature detector response）
    float dresponse;

    // 进化时间（Evolution time）
    float tevolution;

    // 特征点所属的Octave组（Octave of the keypoint）
    float octave;

    // 特征点所属的层级（Sublevel in each octave of the keypoint）
    float sublevel;

    // 特征点的描述向量（Descriptor vector and size）
    std::vector<float> descriptor;
    int descriptor_size;

    // 特征点的主方向（Main orientation）
    float angle;

    // 描述向量类型（Descriptor mode）
    int descriptor_mode;

    // 拉普拉斯标志值（Sign of the laplacian (for faster matching)）
    int laplacian;

    // 进化级别（Evolution Level）
    unsigned int level;

    // Constructor
    Ipoint(void);


    // 按照响应值定义排序规则 （Sort Ipoint by response value）
    bool operator < (const Ipoint& rhs ) const   
    {   
        return dresponse < rhs.dresponse; 
    }
    bool operator > (const Ipoint& rhs ) const   
    {   
        return dresponse > rhs.dresponse; 
    }

};

//*************************************************************************************
//*************************************************************************************

/**
 * Filters for KAZE::Ipoint
 */
void filterDuplicated( std::vector<Ipoint>& keypoints );

void filterRetainBest(std::vector<Ipoint>& keypoints, int n_points);

void filterUnvalidKeypoints( std::vector<Ipoint>& keypoints );

void filterByPixelsMask( std::vector<Ipoint>& keypoints, const cv::Mat& mask );

#endif

