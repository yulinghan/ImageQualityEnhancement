/**
 * @file akaze_demo
 * @brief AKAZE detector + descritpor + BruteForce Matcher + drawing matches with OpenCV functions
 * @author A. Huaman
 * @updated Takahiro Poly Horikawa
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "akaze/akaze_features.h"

using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  {
    readme();
    return -1;
  }

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  {
    std::cerr << " Failed to load images." << std::endl;
    return -1;
  }

  double s, e, t;

  //-- Step 1: Detect the keypoints using SURF Detector
  Ptr<FeatureDetector> detector = FeatureDetector::create("AKAZE");
  // or Ptr<FeatureDetector> detector = new AKAZE();

  s = getTickCount();
  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );
  e = getTickCount();
  t = 1000.0 * (e-s) / getTickFrequency();
  printf("Detect keypoints: %f msec\n", t);

  //-- Step 2: Calculate descriptors (feature vectors)
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("AKAZE");
  // or Ptr<DescriptorExtractor> extractor = new AKAZE();

  s = getTickCount();
  Mat descriptors_1, descriptors_2;

  extractor->compute( img_1, keypoints_1, descriptors_1 );
  extractor->compute( img_2, keypoints_2, descriptors_2 );
  e = getTickCount();
  t = 1000.0 * (e-s) / getTickFrequency();
  printf("Extract descriptors: %f msec\n", t);

  //-- Step 3: Matching descriptor vectors with a brute force matcher
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

  s = getTickCount();
  std::vector< DMatch > matches;
  matcher->match( descriptors_1, descriptors_2, matches );
  e = getTickCount();
  t = 1000.0 * (e-s) / getTickFrequency();
  printf("Match descriptors: %f msec\n", t);

  //-- Draw matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

  //-- Show detected matches
  imshow("Matches", img_matches );

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{
  std::cout << " Usage: ./akaze_demo <img1> <img2>" << std::endl;
}
