
/**
 * @file utils.h
 * @brief Some useful functions
 * @date Dec 29, 2011
 * @author Pablo F. Alcantarilla
 */

#ifndef _UTILS_H_
#define _UTILS_H_

//******************************************************************************
//******************************************************************************

// OPENCV Includes
#include "opencv2/core/core.hpp"

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <math.h>

// Other Includes
#include "kaze_Ipoint.h"

//*************************************************************************************
//*************************************************************************************

// Declaration of Functions
void Compute_min_32F(const cv::Mat &src, float &value);
void Compute_max_32F(const cv::Mat &src, float &value);
void Convert_Scale(cv::Mat &src);
void Copy_and_Convert_Scale(const cv::Mat &src, cv::Mat &dst);
void Draw_Ipoints(cv::Mat &img, const std::vector<Ipoint> &keypoints);
int fRound(float flt);

//*************************************************************************************
//*************************************************************************************

#endif
