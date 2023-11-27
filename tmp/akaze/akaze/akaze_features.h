/**
 * @file akaze_features.h
 * @brief Class that defines cv::AKAZE
 * @author Takahiro Poly Horikawa
 */

#ifndef _AKAZE_FEATURES_H_
#define _AKAZE_FEATURES_H_

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

#include "akaze_config.h"

/*!
 AKAZE features implementation.
 http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html
*/
namespace cv
{
    class CV_EXPORTS_W AKAZE : public Feature2D
    {
    public:

        CV_WRAP explicit AKAZE(int nfeatures = 500, int noctaves = DEFAULT_OCTAVE_MAX, int nlevels = DEFAULT_NSUBLEVELS, float detectorThreshold = DEFAULT_DETECTOR_THRESHOLD, int diffusivityType = DEFAULT_DIFFUSIVITY_TYPE, int descriptorMode = DEFAULT_DESCRIPTOR, int ldbSize = DEFAULT_LDB_DESCRIPTOR_SIZE, int ldbChannels = DEFAULT_LDB_CHANNELS, bool verbosity = DEFAULT_VERBOSITY);

        // returns the descriptor size in bytes
        int descriptorSize() const;

        // returns the descriptor type
        int descriptorType() const;

        // Compute the AKAZE features and descriptors on an image
        void operator()( InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
            OutputArray descriptors, bool useProvidedKeypoints=false ) const;

        // Compute the AKAZE features with mask
        void operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const;

        // Compute the AKAZE features and descriptors on an image without mask
        void operator()(InputArray image, vector<KeyPoint>& keypoints, OutputArray descriptors) const;

        AlgorithmInfo* info() const;

    protected:

        void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

        // !! NOT recommend to use because AKAZE descriptors ONLY work with AKAZE features
        void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

        CV_PROP_RW int nfeatures;
        CV_PROP_RW int noctaves;
        CV_PROP_RW int nlevels;
        CV_PROP_RW float detectorThreshold;
        CV_PROP_RW int diffusivityType;
        CV_PROP_RW int descriptorMode;
        CV_PROP_RW int ldbSize;
        CV_PROP_RW int ldbChannels;
        CV_PROP_RW bool verbosity;
    };

    typedef AKAZE AKazeFeatureDetector;
    //typedef AKAZE AKazeDescriptorExtractor; 	// NOT available because AKAZE descriptors ONLY work with AKAZE features
}

#endif
