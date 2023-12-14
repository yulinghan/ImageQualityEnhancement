
/**
 * @file kaze_features.h
 * @brief Class that defines cv::KAZE
 * @author Ievgen Khvedchenia
 * @update: 2013-03-28 by Yuhua Zou
 */

#ifndef _KAZE_FEATURES_H_
#define _KAZE_FEATURES_H_

#ifdef HAVE_CVCONFIG_H
#include "cvconfig.h"
#endif

#include "opencv2/core/version.hpp"

#if ((CV_MAJOR_VERSION>=2) && (CV_MINOR_VERSION>=4)) 

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"

#else

#include "Minimum_version_2.4.0_please_upgrade_your_OpenCV"

#endif

#include <algorithm>

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/features2d/features2d_tegra.hpp"
#endif

#include "kaze_config.h"

/*!
 KAZE features implementation.
 http://www.robesafe.com/personal/pablo.alcantarilla/papers/Alcantarilla12eccv.pdf
*/
namespace cv
{
    class CV_EXPORTS_W KAZE : public Feature2D
    {
    public:

        CV_WRAP explicit KAZE( int nfeatures = 0, int noctaves = 4, int nlevels = 4, float detectorThreshold = 0.001,
            int diffusivityType = 1, int descriptorMode = 1, bool extendDescriptor = false, bool uprightOrient = false, bool verbosity = false );
        
        KAZE(toptions &_options);

        // returns the descriptor size in bytes
        int descriptorSize() const;

        // returns the descriptor type
        int descriptorType() const;

        // Compute the KAZE features and descriptors on an image
        void operator()( InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
            OutputArray descriptors, bool useProvidedKeypoints=false ) const;

        // Compute the KAZE features with mask
        void operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const;

        // Compute the KAZE features and descriptors on an image without mask
        void operator()(InputArray image, vector<KeyPoint>& keypoints, OutputArray descriptors) const;

        //AlgorithmInfo* info() const;

    protected:

        void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

        // !! NOT recommend to use because KAZE descriptors ONLY work with KAZE features
        void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

        CV_PROP_RW int nfeatures;

    private:
        toptions options;
    };

    typedef KAZE KazeFeatureDetector;
    //typedef KAZE KazeDescriptorExtractor; 	// NOT available because KAZE descriptors ONLY work with KAZE features
}

#endif
