/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** Authors: Ievgen Khvedchenia */
/** Update: 2013-03-28 by Yuhua Zou*/

#include <iterator>
#include "kaze_features.h"
#include "kaze.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEGREE_TO_RADIAN(x) ((x) * CV_PI / 180.0)
#define RADIAN_TO_DEGREE(x) ((x) * 180.0 / CV_PI)

namespace cv
{
    /***
     *    Convertions between cv::Keypoint and KAZE::Ipoint
     */
    static inline void convertPoint(const cv::KeyPoint& kp, Ipoint& aux)
    {
        aux.xf = kp.pt.x;
        aux.yf = kp.pt.y;
        aux.x = fRound(aux.xf);
        aux.y = fRound(aux.yf);

        //cout << "SURF size: " << kpts_surf1_[i].size*.5 << endl;
        aux.octave = kp.octave;

        // Get the radius for visualization
        aux.scale = kp.size*.2;    // Updated by Yuhua Zou
        aux.angle = DEGREE_TO_RADIAN(kp.angle);

        //aux.descriptor_size = 64;
    }

    static inline void convertPoint(const Ipoint& src, cv::KeyPoint& kp)
    {
        kp.pt.x = src.xf;
        kp.pt.y = src.yf;

        kp.angle    = RADIAN_TO_DEGREE(src.angle);
        kp.response = src.dresponse;

        kp.octave = src.octave;    
        kp.size = src.scale*5.0;    // Updated by Yuhua Zou
    }

    /***
     *    Implementation of cv::KAZE
     */
    KAZE::KAZE( int nfeatures /* = 1000 */, int noctaves /* = 2 */, 
        int nlevels /* = 4 */, float detectorThreshold /* = 0.001 */, 
        int diffusivityType /* = 1 */, int descriptorMode /* = 1 */, 
        bool extendDescriptor /* = false */, bool uprightOrient /* = false */, 
        bool verbosity /* = false */ )
    {
        options.nfeatures = nfeatures;
        options.omax = noctaves;
        options.nsublevels = nlevels;
        options.dthreshold = detectorThreshold;
        options.diffusivity = diffusivityType;
        options.descriptor = descriptorMode;
        options.extended = extendDescriptor;
        options.upright = uprightOrient;
        options.verbosity = verbosity;
    }

    KAZE::KAZE(toptions &_options)
    {
        options = _options;
    }

    int KAZE::descriptorSize() const
    {
        return options.extended ? 128 : 64;
    }

    int KAZE::descriptorType() const
    {
        return CV_32F;
    }

    void KAZE::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
        OutputArray _descriptors, bool useProvidedKeypoints) const
    {

        bool do_keypoints = !useProvidedKeypoints;
        bool do_descriptors = _descriptors.needed();

        if( (!do_keypoints && !do_descriptors) || _image.empty() )
            return;
        
        cv::Mat img1_8, img1_32;

        // Convert to gray scale iamge and float image
        if (_image.getMat().channels() == 3)
            cv::cvtColor(_image, img1_8, CV_RGB2GRAY);
        else
            _image.getMat().copyTo(img1_8);

        img1_8.convertTo(img1_32, CV_32F, 1.0/255.0,0);

        // Construct KAZE
        toptions opt = options;
        opt.img_width = img1_32.cols;
        opt.img_height = img1_32.rows;

        ::KAZE kazeEvolution(opt);

        // Create nonlinear scale space
        kazeEvolution.Create_Nonlinear_Scale_Space(img1_32);        

        // Feature detection
        std::vector<Ipoint> kazePoints;

        if (do_keypoints)
        {
            kazeEvolution.Feature_Detection(kazePoints);
            filterDuplicated(kazePoints);

            if (!_mask.empty())
            {
                filterByPixelsMask(kazePoints, _mask.getMat());
            }

            if (opt.nfeatures > 0)
            {
                filterRetainBest(kazePoints, opt.nfeatures);
            }
            
        }
        else
        {
            kazePoints.resize(_keypoints.size());

            #pragma omp parallel for
            for (int i = 0; i < kazePoints.size(); i++)
            {
                convertPoint(_keypoints[i], kazePoints[i]);    
            }
        }
        
        // Descriptor caculation
        if (do_descriptors)
        {
            kazeEvolution.Feature_Description(kazePoints);

            cv::Mat& descriptors = _descriptors.getMatRef();
            descriptors.create(kazePoints.size(), descriptorSize(), descriptorType());

            for (size_t i = 0; i < kazePoints.size(); i++)
            {
                std::copy(kazePoints[i].descriptor.begin(), kazePoints[i].descriptor.end(), (float*)descriptors.row(i).data);
            }
        }

        // Transfer from KAZE::Ipoint to cv::KeyPoint
        if (do_keypoints)
        {
            _keypoints.resize(kazePoints.size());

            #pragma omp parallel for
            for (int i = 0; i < kazePoints.size(); i++)
            {
                convertPoint(kazePoints[i], _keypoints[i]);
            }
        }
        
    }

    void KAZE::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints ) const
    {
        (*this)(image, mask, keypoints, noArray(), false);
    }

    void KAZE::operator()(InputArray image, vector<KeyPoint>& keypoints, OutputArray descriptors) const
    {
        (*this)(image, noArray(), keypoints, descriptors, false);
    }

    void KAZE::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
    {
        (*this)(image, mask, keypoints, noArray(), false);
    }

    void KAZE::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
    {
        (*this)(image, Mat(), keypoints, descriptors, false);
    }

}
