//=============================================================================
//
// Ipoint.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 21/01/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file Ipoint.cpp
 * @brief Class that defines a point of interest
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 * @update: 2013-03-28 by Yuhua Zou
 */

#include "kaze_Ipoint.h"

//*******************************************************************************
//*******************************************************************************

/**
 * @brief Ipoint default constructor
 */
Ipoint::Ipoint(void)
{
    xf = yf = 0.0;
    x = y = 0;
    scale = 0.0;
    dresponse = 0.0;
    tevolution = 0.0;
    octave = 0.0;
    sublevel = 0.0;
    descriptor_size = 0;
    descriptor_mode = 0;
    laplacian = 0;
    level = 0;
}

//*******************************************************************************
//*******************************************************************************

/******************Updated by Yuhua Zou begin************************************/

/***
    *    Filters for KAZE Ipoint
    */
class MaskPredicate
{
public:
    MaskPredicate( const cv::Mat& _mask ) : mask(_mask) {}
    bool operator() (const Ipoint& key_pt) const
    {
        return mask.at<uchar>( (int)(key_pt.yf + 0.5f), (int)(key_pt.xf + 0.5f) ) == 0;
    }

private:
    const cv::Mat mask;
    MaskPredicate& operator = (const MaskPredicate&);
};
    
void filterByPixelsMask( std::vector<Ipoint>& keypoints, const cv::Mat& mask )
{
    if( mask.empty() )
        return;

    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
}
    
class ResponsePredicate
{
public:
    ResponsePredicate() {}
    bool operator() (const Ipoint& key_pt) const
    {
        return key_pt.dresponse == 0;
    }

private:
    ResponsePredicate& operator=(const ResponsePredicate&);
};

void filterUnvalidKeypoints( std::vector<Ipoint>& keypoints )
{
    if( keypoints.empty() )
        return;

    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), ResponsePredicate()), keypoints.end());
}

void filterDuplicated( std::vector<Ipoint>& keypoints )
{
    int i, j, n = (int)keypoints.size();
    int esigma = 0, level = 0;
    float dist = 0.0;

    for (i = 0; i < n; i++)
    {
        if (keypoints[i].dresponse == 0)
            continue;

        level = keypoints[i].level;
        esigma = keypoints[i].sigma_size;
        esigma *= esigma;

        for (j = 0; j < n; j++)
        {
            if ( (j != i) && (keypoints[j].dresponse == 0) &&
                ( keypoints[j].level == level || keypoints[j].level == level+1 || keypoints[j].level == level-1 ))
            {                            
                dist = pow(keypoints[j].xf-keypoints[i].xf,2)+pow(keypoints[j].yf-keypoints[i].yf,2);
                if( dist < esigma )
                {
                    if( keypoints[j].dresponse > keypoints[i].dresponse )
                        keypoints[i].dresponse = 0;
                    else
                        keypoints[j].dresponse = 0;

                    break;
                }
            }
        }            
    }

    filterUnvalidKeypoints(keypoints);
}
    
void filterRetainBest(std::vector<Ipoint>& keypoints, int n_points)
{
    //this is only necessary if the keypoints size is greater than the number of desired points.
    if( n_points > 0 && keypoints.size() > (size_t)n_points )
    {
        std::sort(keypoints.begin(), keypoints.end(), std::greater<Ipoint>());
        keypoints.resize(n_points);
    }
}

/******************Updated by Yuhua Zou end************************************/
