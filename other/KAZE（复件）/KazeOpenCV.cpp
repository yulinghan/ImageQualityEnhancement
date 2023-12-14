/**
 * @file KazeOpenCV.cpp
 * @brief Sample code showing how to match images using KAZE features
 * @date March 28, 2013
 * @author Yuhua Zou (yuhuazou@gmail.com)
 */

#include "predep.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

// !! Please enable /openmp in your project configurations (in /C++/Language) in Visual Studio
//    If you have installed and included Boost in your project, 
//    please set 'HAVE_BOOST_THREADING' to 1 in ./KAZE/kaze_config.h to enable Boost-based multi-threading
#include "KAZE/kaze_features.h"

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )
#pragma comment( lib, cvLIB("flann") )
#pragma comment( lib, cvLIB("features2d") )
#pragma comment( lib, cvLIB("calib3d") )

// Define 'USE_SIFT' to use SIFT keypoints instead of KAZE for comparation 
#define USE_SIFT0    

#ifdef USE_SIFT
#include "opencv2/nonfree/features2d.hpp"
#pragma comment( lib, cvLIB("nonfree") )
#endif


using namespace cv;
using namespace std;

// @brief Show text in the upper left corner of the image
void showText(cv::Mat& img, string text)
{
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.5;
    int fontThickness = 3;

    // get text size
    int textBaseline=0;
    cv::Size textSize = cv::getTextSize(text, fontFace,
        fontScale, fontThickness, &textBaseline);
    textBaseline += fontThickness;

    // put the text at upper right corner
    //cv::Point textOrg((img.cols - textSize.width - 10), textSize.height + 10);
    cv::Point textOrg(10, textSize.height + 10); // upper left corner

    // draw the box
    rectangle(img, textOrg + cv::Point(0, textBaseline),
        textOrg + cv::Point(textSize.width, -textSize.height-10),
        cv::Scalar(50,50,50), -1);

    // then put the text itself
    putText(img, text, textOrg, fontFace, fontScale,
        cv::Scalar(0,0,255), fontThickness, 8);
}

// @brief Find homography and inliers
bool findHomography( const vector<KeyPoint>& source, const vector<KeyPoint>& result, const vector<DMatch>& input, vector<DMatch>& inliers, cv::Mat& homography)
{
    if (input.size() < 4)
        return false;

    const int pointsCount = input.size();
    const float reprojectionThreshold = 3;

    //Prepare src and dst points
    std::vector<cv::Point2f> srcPoints, dstPoints;    
    for (int i = 0; i < pointsCount; i++)
    {
        srcPoints.push_back(source[input[i].queryIdx].pt);
        dstPoints.push_back(result[input[i].trainIdx].pt);
    }

    // Find homography using RANSAC algorithm
    std::vector<unsigned char> status;
    homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, status);

    // Warp dstPoints to srcPoints domain using inverted homography transformation
    std::vector<cv::Point2f> srcReprojected;
    cv::perspectiveTransform(dstPoints, srcReprojected, homography.inv());

    // Pass only matches with low reprojection error (less than reprojectionThreshold value in pixels)
    inliers.clear();
    for (int i = 0; i < pointsCount; i++)
    {
        cv::Point2f actual = srcPoints[i];
        cv::Point2f expect = srcReprojected[i];
        cv::Point2f v = actual - expect;
        float distanceSquared = v.dot(v);

        if (/*status[i] && */distanceSquared <= reprojectionThreshold * reprojectionThreshold)
        {
            inliers.push_back(input[i]);
        }
    }

    return inliers.size() >= 4;
}

// @brief Use BFMatcher to match descriptors
void bfMatch( Mat& descriptors_1, Mat& descriptors_2, vector<DMatch>& good_matches, bool filterMatches = true )
{
    //-- Matching descriptor vectors using Brute-Force matcher
    cout << "--> Use BFMatcher..." << endl;
    BFMatcher matcher(cv::NORM_L2, true);
    vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    if (!filterMatches)
    {
        good_matches = matches;
    } 
    else
    {
        double max_dist = 0, min_dist = 100, thresh = 0;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < matches.size(); i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        //thresh = MAX(2*min_dist, min_dist + 0.5*(max_dist - min_dist));
        thresh = 2*min_dist;

        //-- Find initial good matches (i.e. whose distance is less than 2*min_dist )
        for( int i = 0; i < matches.size(); i++ )
        { 
            if( matches[i].distance < thresh )    
            { 
                good_matches.push_back( matches[i]); 
            }
        }
    }
}

// @brief Use FlannBasedMatcher to match descriptors
void flannMatch( Mat& descriptors_1, Mat& descriptors_2, vector<DMatch>& good_matches, bool filterMatches = true )
{
    cout << "--> Use FlannBasedMatcher..." << endl;
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    if (!filterMatches)
    {
        good_matches = matches;
    } 
    else
    {
        double max_dist = 0, min_dist = 100, thresh = 0;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < matches.size(); i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        //thresh = MAX(2*min_dist, min_dist + 0.5*(max_dist - min_dist));
        thresh = 2*min_dist;

        //-- Find initial good matches (i.e. whose distance is less than 2*min_dist )
        for( int i = 0; i < matches.size(); i++ )
        { 
            if( matches[i].distance < thresh )    
            { 
                good_matches.push_back( matches[i]); 
            }
        }
    }
}

// @brief Use FlannBasedMatcher with knnMatch to match descriptors
void knnMatch( Mat& descriptors_1, Mat& descriptors_2, vector<DMatch>& good_matches )
{
    cout << "--> Use knnMatch..." << endl;
    vector<vector<DMatch> > knMatches;
    FlannBasedMatcher matcher;
    int k = 2;
    float maxRatio = 0.75;

    matcher.knnMatch(descriptors_1, descriptors_2, knMatches, k);

    good_matches.clear();

    for (size_t i=0; i< knMatches.size(); i++)
    {
        const cv::DMatch& best = knMatches[i][0];
        const cv::DMatch& good = knMatches[i][1];

        //if (best.distance <= good.distance) continue;

        float ratio = (best.distance / good.distance);
        if (ratio <= maxRatio)
        {
            good_matches.push_back(best);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// @brief Main function
int main(int argc, char** argv)
{
    if (argc < 3)
        return 0;

    ////////////////////////////////////////////////////////////////////////////////////
    //-- Load object image
    Mat img_1 = imread(argv[1]); if (img_1.empty()) return -1;

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    bool doDrawKeypoint = true;
    float beta = 1;
    int nMatches = argc - 2;
    Mat imgMatches;
    int roiHeight = (int)(img_1.rows*beta);


    //-- Construct feature engine for object image
#ifdef USE_SIFT
    cv::SiftFeatureDetector detector_1;
    cv::SiftDescriptorExtractor extractor_1;
#else
    toptions opt;
    opt.omax = 2;
    //opt.nfeatures = 1000;
    //opt.verbosity = true;
    KAZE detector_1(opt), detector_2(opt);
#endif

    double tkaze = 0.0;
    int64 t1 = cv::getTickCount(), t2 = 0;

    //-- Detect keypoints and calculate descriptors
#ifdef USE_SIFT
    detector_1.detect(img_1, keypoints_1);
    extractor_1.compute(img_1,keypoints_1,descriptors_1);
#else
    detector_1(img_1, keypoints_1, descriptors_1);
#endif

    t2 = cv::getTickCount();
    tkaze = 1000.0 * (t2 - t1) / cv::getTickFrequency();

    cout << "\n-- Detection time (ms): " << tkaze << endl;
    printf("-- Keypoint number of img_1 : %d \n", keypoints_1.size() );

    //return 0;

    //-- Draw Keypoints
    showText(img_1, "Image #1");
    if (doDrawKeypoint)
    {
        drawKeypoints(img_1, keypoints_1, img_1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    for (int k = 2; k < argc; k++)
    {
        Mat img_2 = imread(argv[k]); if (img_2.empty()) continue;
        
        ////////////////////////////////////////////////////////////////////////////////////
        t1 = cv::getTickCount();

        //-- Detect keypoints and calculate descriptors 
#ifdef USE_SIFT
        detector_2.detect(img_2, keypoints_2);
        extractor_2.compute(img_2,keypoints_2,descriptors_2);
#else
        detector_2(img_2, keypoints_2, descriptors_2);
#endif

        t2 = cv::getTickCount();
        tkaze = 1000.0 * (t2 - t1) / cv::getTickFrequency();

        cout << "\n-- Detection time (ms): " << tkaze << endl;
        printf("-- Keypoint number of img_2 : %d \n", keypoints_2.size() );

        if (keypoints_1.size() < 4 || keypoints_2.size() < 4)
            continue;

        ////////////////////////////////////////////////////////////////////////////////////
        //-- Matching Keypoints
        cout << "-- Computing homography (RANSAC)..." << endl;
        vector<DMatch> matches, inliers;
        Mat H;
        bool filterMatches = true;

        bfMatch(descriptors_1, descriptors_2, matches, filterMatches);
        if (!::findHomography(keypoints_1, keypoints_2, matches, inliers, H))
        {
            matches.clear();
            flannMatch(descriptors_1, descriptors_2, matches, filterMatches);
            if (!::findHomography(keypoints_1, keypoints_2, matches, inliers, H))
            {
                matches.clear();
                knnMatch(descriptors_1, descriptors_2, matches);
                if (!::findHomography(keypoints_1, keypoints_2, matches, inliers, H))
                {
                    inliers.clear();
                    H = Mat();
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////
        //-- Draw Keypoints
        char tiImg[20];
        sprintf_s(tiImg, "Image #%d", k);
        showText(img_2, tiImg);
        if (doDrawKeypoint)
        {
            drawKeypoints(img_2, keypoints_2, img_2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }

        //-- Draw inliers
        Mat imgMatch;
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
            inliers, imgMatch, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        printf("-- Number of Matches : %d \n", matches.size() );
        printf("-- Number of Inliers : %d \n", inliers.size() );
        printf("-- Match rate : %f \n", inliers.size() / (float)matches.size() );

        //-- Localize the object
        //-- Get the corners from the image_1 ( the object to be "detected" )
        vector<Point2f> obj_corners;
        obj_corners.push_back( Point2f(0,0) );
        obj_corners.push_back( Point2f(img_1.cols,0) );
        obj_corners.push_back( Point2f(img_1.cols,img_1.rows) );
        obj_corners.push_back( Point2f(0,img_1.rows) );

        if (!H.empty())
        {
            vector<Point2f> scene_corners;
            perspectiveTransform(obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            int npts = scene_corners.size();
            for (int i=0; i<npts; i++)
                line( imgMatch, scene_corners[i] + Point2f( img_1.cols, 0), 
                scene_corners[(i+1)%npts] + Point2f( img_1.cols, 0), Scalar(0,70*i,255), 6 );
        }

        //-- Combine all matches
        if (imgMatches.empty())
            imgMatches = Mat( img_1.rows*nMatches*beta, (img_1.cols+img_2.cols)*beta, CV_8UC3 );
        Rect roi = Rect(0, (k-2)*roiHeight, imgMatches.cols, roiHeight);
        Mat imgRoi = imgMatches(roi);
        int step = imgMatches.cols/40, step2 = step*2;
        resize(imgMatch, imgRoi, imgRoi.size());
        if (k==6) continue;
        for (int d = 0; d < imgMatches.cols; d += step2)
            line(imgMatches, Point(d,(k-1)*roiHeight), Point(d+step,(k-1)*roiHeight), Scalar(255,255,0), 3, 8);
    }
    // End for
    int step = imgMatches.rows/40, step2 = step*2;
    for (int d = 0; d < imgMatches.rows; d += step2)
        line(imgMatches, Point(img_1.cols*beta,d), Point(img_1.cols*beta,d+step), Scalar(255,255,0), 2, 8);

    ////////////////////////////////////////////////////////////////////////////////////
    // Save match result 
#ifdef USE_SIFT
    imwrite("../../blog_pics/Sift_.png", imgMatches);
#else
    imwrite("../../blog_pics/Kaze_.png", imgMatches);
#endif

    // Rotating the image by -90 degree
    if (imgMatches.rows > imgMatches.cols)
    {
        flip(imgMatches.t(), imgMatches, 0);
    }

    // Show detected matches
    namedWindow("Matches",CV_WINDOW_NORMAL);
    imshow( "Matches", imgMatches );

    waitKey(0);
    destroyAllWindows();

    return 0;
}
