
//=============================================================================
//
// utils.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 29/12/2011
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file utils.cpp
 * @brief Some useful functions
 * @date Dec 29, 2011
 * @author Pablo F. Alcantarilla
 */

#include "kaze_utils.h"

// Namespaces
using namespace std;

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the minimum value of a float image
 * @param src Input image
 * @param value Minimum value
 */
void Compute_min_32F(const cv::Mat &src, float &value)
{
   float aux = 1000.0;
   
   for( int i = 0; i < src.rows; i++ )
   {
       for( int j = 0; j < src.cols; j++ )
       {
           if( src.at<float>(i,j) < aux )
           {
               aux = src.at<float>(i,j);
           }
       }
   }    
   
   value = aux;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the maximum value of a float image
 * @param src Input image
 * @param value Maximum value
 */
void Compute_max_32F(const cv::Mat &src, float &value)
{
   float aux = 0.0;

   for( int i = 0; i < src.rows; i++ )
   {
       for( int j = 0; j < src.cols; j++ )
       {
           if( src.at<float>(i,j) > aux )
           {
               aux = src.at<float>(i,j);
           }
       }
   }    
  
   value = aux;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function converts the scale of the input image prior to visualization
 * @param src Input/Output image
 * @param value Maximum value
 */
void Convert_Scale(cv::Mat &src)
{
   float min_val = 0, max_val = 0;

   Compute_min_32F(src,min_val);

   src = src - min_val;

   Compute_max_32F(src,max_val);
   src = src / max_val;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function copies the input image and converts the scale of the copied
 * image prior visualization
 * @param src Input image
 * @param dst Output image
 */
void Copy_and_Convert_Scale(const cv::Mat &src, cv::Mat dst)
{
   float min_val = 0, max_val = 0;

   src.copyTo(dst);
   Compute_min_32F(dst,min_val);

   dst = dst - min_val;

   Compute_max_32F(dst,max_val);
   dst = dst / max_val;
   
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function draws a vector of Ipoints
 * @param img Input/Output Image
 * @param dst Vector of keypoints
 */
void Draw_Ipoints(cv::Mat &img, const std::vector<Ipoint> &keypoints)
{
    int x = 0, y = 0;
    float s = 0.0;
    
    for( unsigned int i = 0; i < keypoints.size(); i++ )
    {
        x = keypoints[i].x;
        y = keypoints[i].y;
        s = keypoints[i].scale*2.0;
    
        // Draw a circle centered on the interest point
        cv::circle(img,cv::Point(x,y),s,cv::Scalar(255,0,0),1);
        cv::circle(img,cv::Point(x,y),1.0,cv::Scalar(0,255,0),-1);
    }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This funtion rounds float to nearest integer
 * @param flt Input float
 * @return dst Nearest integer
 */
int fRound(float flt)
{
  return (int)(flt+0.5);
}
