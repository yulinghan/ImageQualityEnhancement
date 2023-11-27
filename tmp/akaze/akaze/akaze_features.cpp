/**
 * @file akaze_features.cpp
 * @brief Class that defines cv::AKAZE
 * @author Takahiro Poly Horikawa
 */

#include "akaze_features.h"
#include "AKAZE.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv
{
    /***
     *    Implementation of cv::AKAZE
     */
    AKAZE::AKAZE(int _nfeatures, int _noctaves, int _nlevels, float _detectorThreshold, int _diffusivityType, int _descriptorMode, int _ldbSize, int _ldbChannels, bool _verbosity) :
        nfeatures(_nfeatures), noctaves(_noctaves), nlevels(_nlevels), detectorThreshold(_detectorThreshold), diffusivityType(_diffusivityType), descriptorMode(_descriptorMode), ldbSize(_ldbSize), ldbChannels(_ldbChannels), verbosity(_verbosity)
    {
    }

    int AKAZE::descriptorSize() const
    {
        // Allocate memory for the matrix with the descriptors
        if (descriptorMode < MLDB_UPRIGHT) {
            return 64;
        }
        else {
            // We use the full length binary descriptor -> 486 bits
            if (ldbSize == 0) {
                int t = (6+36+120)*ldbChannels;
                return ceil(t/8.);
            }
            else {
                // We use the random bit selection length binary descriptor
                return ceil(ldbSize/8.);
            }
        }
    }

    int AKAZE::descriptorType() const
    {
        if (descriptorMode < MLDB_UPRIGHT) {
            return CV_32F;
        } else {
            return CV_8U;
        }
    }

    void AKAZE::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
        OutputArray _descriptors, bool useProvidedKeypoints) const
    {

        bool do_keypoints = !useProvidedKeypoints;
        bool do_descriptors = _descriptors.needed();

        if( (!do_keypoints && !do_descriptors) || _image.empty() )
            return;

        cv::Mat img1_32;
        Mat img = _image.getMat();
        img.convertTo(img1_32, CV_32F, 1.0/255.0 ,0);

        AKAZEOptions opt;
        opt.img_width = img.cols;
        opt.img_height = img.rows;
        opt.omax = noctaves;
        opt.dthreshold = detectorThreshold;
        opt.nsublevels = nlevels;
        opt.diffusivity = diffusivityType;
        opt.descriptor = descriptorMode;
        opt.descriptor_size = ldbSize;
        opt.descriptor_channels = ldbChannels;
        opt.verbosity = verbosity;

        ::AKAZE evolution(opt);

        evolution.Create_Nonlinear_Scale_Space(img1_32);

        if (do_keypoints)
        {
            _keypoints.clear();
            evolution.Feature_Detection(_keypoints);

            if (!_mask.empty())
            {
                KeyPointsFilter::runByPixelsMask(_keypoints, _mask.getMat());
            }

            if (nfeatures > 0)
            {
                KeyPointsFilter::retainBest(_keypoints, nfeatures);
            }
        }

        if (do_descriptors)
        {
            //cv::KeyPointsFilter::runByImageBorder(_keypoints, cv::Size(opt.img_width, opt.img_height), 40);
            cv::Mat& descriptors = _descriptors.getMatRef();
            evolution.Compute_Descriptors(_keypoints, descriptors);
        }
    }

    void AKAZE::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints ) const
    {
        (*this)(image, mask, keypoints, noArray(), false);
    }

    void AKAZE::operator()(InputArray image, vector<KeyPoint>& keypoints, OutputArray descriptors) const
    {
        (*this)(image, noArray(), keypoints, descriptors, false);
    }

    void AKAZE::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
    {
        (*this)(image, mask, keypoints, noArray(), false);
    }

    void AKAZE::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
    {
        (*this)(image, Mat(), keypoints, descriptors, false);
    }

	CV_INIT_ALGORITHM(AKAZE, "Feature2D.AKAZE",
                  obj.info()->addParam(obj, "nfeatures", obj.nfeatures);
                  obj.info()->addParam(obj, "noctaves", obj.noctaves);
                  obj.info()->addParam(obj, "nlevels", obj.nlevels);
                  obj.info()->addParam(obj, "detectorThreshold", obj.detectorThreshold);
                  obj.info()->addParam(obj, "diffusivityType", obj.diffusivityType);
                  obj.info()->addParam(obj, "descriptorMode", obj.descriptorMode);
                  obj.info()->addParam(obj, "ldbSize", obj.ldbSize);
                  obj.info()->addParam(obj, "ldbChannels", obj.ldbChannels);
                  obj.info()->addParam(obj, "verbosity", obj.verbosity));
}
