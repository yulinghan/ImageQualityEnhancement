#include "SGridTracker.h"
#include <cstdio>

using namespace std;
using namespace cv;

GridTracker::GridTracker() {

}

bool GridTracker::maskPoint(float x, float y)
{
    if (curMask.at<unsigned char>(cv::Point(x, y)) == 0)// 0 indicates that this pixel is in the mask, thus is not useable for new features of OF results
        return 1;// means that this feature should be killed
    cv::rectangle(curMask, cv::Point(int(x - MASK_RADIUS / 2 + .5), int(y - MASK_RADIUS / 2 + .5)), cv::Point(int(x + MASK_RADIUS / 2 + .5), int(y + MASK_RADIUS / 2 + .5)), cv::Scalar(0), -1);//define a new image patch
    return 0;// means that this feature can be retained
};

bool GridTracker::trackerInit(cv::Mat& im)
{
    curMask = Mat(im.rows, im.cols, CV_8UC1, Scalar(255));

    /*curMask.create(im.rows,im.cols,CV_8UC1);
    for(int i=0;i<im.rows;i++){
        for(int j=0;j<im.cols;j++){
            curMask.at<uchar>(i,j)=255;
        }
    }*/

    numActiveTracks = 0;
    TRACKING_HSIZE = 8;
    LK_PYRAMID_LEVEL = 4;
    MAX_ITER = 10;
    ACCURACY = 0.1;
    LAMBDA = 0.0;

    //overflow = 0;

    hgrids.x = GRIDSIZE;
    hgrids.y = GRIDSIZE;

    usableFrac = 0.02;

    MaxTracks = MAXTRACKS;

    minAddFrac = 0.1;
    minToAdd = minAddFrac * MaxTracks;

    //upper limit of features of each grid
    fealimitGrid = floor((float)MaxTracks / (float)(hgrids.x*hgrids.y));

    lastNumDetectedGridFeatures.resize((hgrids.x*hgrids.y), 0);

    DETECT_GAIN = 10;

    for (int i = 0;i < (hgrids.x*hgrids.y);i++)
    {
        hthresholds.push_back(20);
#if CV_MAJOR_VERSION < 3
		cv::FastFeatureDetector detectorInit(hthresholds[i], true);
		detector.push_back(detectorInit);
#else
        detector.push_back(cv::FastFeatureDetector::create(hthresholds[i], true));
#endif
        feanumofGrid.push_back(0);
    }

    cv::buildOpticalFlowPyramid(im, prevPyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);

    Update(im, im);

    return 1;
}

bool GridTracker::Update(cv::Mat& im0, cv::Mat& im1)
{
    //clear the mask to be white everywhere  
    curMask.setTo(Scalar(255));
    /*for(int i=0;i<curMask.rows;i++){
        for(int j=0;j<curMask.cols;j++){
            curMask.at<uchar>(i,j)=255;
        }
    }*/

    //num of feas from last frame
    numActiveTracks = allFeas.size();


    //do optical flow if there are feas from last frame
    if (numActiveTracks > 0)
    {
        vector<uchar> status(allFeas.size(), 1);
        vector<float> error(allFeas.size(), -1);

        //image pyramid of curr frame
        std::vector<cv::Mat> nextPyr;
        cv::buildOpticalFlowPyramid(im1, nextPyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);

        // perform LK tracking from OpenCV, parameters matter a lot
        points1 = allFeas;

        preFeas.clear();
        trackedFeas.clear();



        cv::calcOpticalFlowPyrLK(prevPyr,
            nextPyr,
            cv::Mat(allFeas),
            cv::Mat(points1),
            cv::Mat(status),// '1' indicates successfull OF from points0
            cv::Mat(error),
            Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1),//size of searching window for each Pyramid level
            LK_PYRAMID_LEVEL,// now is 4, the maximum Pyramid levels
            TermCriteria(TermCriteria::COUNT | TermCriteria::EPS,// "type", this means that both termcriteria work here
                MAX_ITER,
                ACCURACY),
            1,//enables optical flow initialization
            LAMBDA);//minEigTheshold

        //renew prevPyr
        //prevPyr.swap(nextPyr);

        //clear feature counting for each grid
        for (int k = 0;k < (hgrids.x*hgrids.y);k++)
        {
            feanumofGrid[k] = 0;
        }

        // 	  overflow = 0;

        for (size_t i = 0; i < points1.size(); i++)
        {
            if (status[i] && points1[i].x > usableFrac*im1.cols && points1[i].x < (1.0 - usableFrac)*im1.cols && points1[i].y > usableFrac*im1.rows && points1[i].y < (1.0 - usableFrac)*im1.rows)
            {
                bool shouldKill = maskPoint(points1[i].x, points1[i].y);

                if (shouldKill)
                {
                    numActiveTracks--;
                }
                else
                {
                    preFeas.push_back(allFeas[i]);
                    trackedFeas.push_back(points1[i]);

                    int hgridIdx =
                        (int)(floor((float)(points1[i].x) / (float)((float)(im1.size().width) / (float)(hgrids.x)))
                            + hgrids.x * floor((float)(points1[i].y) / (float)((float)(im1.size().height) / (float)(hgrids.y))));

                    feanumofGrid[hgridIdx]++;
                }
            }
            else
            {
                numActiveTracks--;
            }
        }
    }
    else
    {
        //clear feature counting for each grid
        for (int k = 0;k < (hgrids.x*hgrids.y);k++)
        {
            feanumofGrid[k] = 0;
        }

        // 	  overflow = MaxTracks;
    }

    allFeas = trackedFeas;

    int ntoadd = MaxTracks - numActiveTracks;



    if (ntoadd > minToAdd)
    {
        //unusedRoom sum
        unusedRoom = 0;

        //hungry sum
        gridsHungry = 0;

        //hungry grids
        vector<pair<int, float> > hungryGrid;

        //room for adding featurs to each grid
        int room = 0;

        //the hungry degree of a whole frame 
        int hungry = 0;

        //set a specific cell as ROI
        Mat sub_image;

        //set the corresponding mask for the previously choosen cell
        Mat sub_mask;

        //keypoints detected from each grids
        vector<vector<cv::KeyPoint> > sub_keypoints;
        sub_keypoints.resize(hgrids.x * hgrids.y);

        //patch for computing variance
        cv::Mat patch;
        int midGrid = floor((hgrids.x*hgrids.y - 1) / 2.0);

        //the first round resampling on each grid
        for (int q = 0; q < hgrids.x*hgrids.y && numActiveTracks < MaxTracks; q++)
        {

            int i = q;
            if (q == 0)
                i = midGrid;
            if (q == midGrid)
                i = 0;

            room = fealimitGrid - feanumofGrid[i];
            if (room > fealimitGrid*minAddFrac)
            {
                //rowIndx for cells
                int celly = i / hgrids.x;

                //colIndx for cells
                int cellx = i - celly * hgrids.x;

                //rowRang for pixels
                Range row_range((celly*im1.rows) / hgrids.y, ((celly + 1)*im1.rows) / hgrids.y);

                //colRange for pixels
                Range col_range((cellx*im1.cols) / hgrids.x, ((cellx + 1)*im1.cols) / hgrids.x);

                sub_image = im1(cv::Rect(col_range.start,//min_x
                    row_range.start,//min_y
                    col_range.size(),//length_x
                    row_range.size()//length_y
                ));

                sub_mask = curMask(cv::Rect(col_range.start,
                    row_range.start,
                    col_range.size(),
                    row_range.size()
                ));


                float lastP = ((float)lastNumDetectedGridFeatures[i] - (float)15 * room) / ((float)15 * room);
#if CV_MAJOR_VERSION < 3
				float newThresh = detector[i].getDouble("threshold");
#else
				float newThresh = detector[i]->getThreshold();
#endif
                newThresh = newThresh + ceil(DETECT_GAIN*lastP);

                if (newThresh > 200)
                    newThresh = 200;
                if (newThresh < 5.0)
                    newThresh = 5.0;
#if CV_MAJOR_VERSION < 3
                detector[i].set("threshold", newThresh);
                //detect keypoints in this cell
                detector[i].detect(sub_image, sub_keypoints[i], sub_mask);
#else
				detector[i]->setThreshold(newThresh);
				//detect keypoints in this cell
				detector[i]->detect(sub_image, sub_keypoints[i], sub_mask);
#endif

                lastNumDetectedGridFeatures[i] = sub_keypoints[i].size();
                KeyPointsFilter::retainBest(sub_keypoints[i], 2 * fealimitGrid);

                //sort features
                std::sort(sub_keypoints[i].begin(), sub_keypoints[i].end(), rule);

                //for each feature ...
                std::vector<cv::KeyPoint>::iterator it = sub_keypoints[i].begin(), end = sub_keypoints[i].end();
                int n = 0;



                //first round
                for (; n < room && it != end && numActiveTracks < MaxTracks; ++it)
                {
                    //transform grid based position to image based position
                    it->pt.x += col_range.start;
                    it->pt.y += row_range.start;

                    //check is features are being too close
                    if (curMask.at<unsigned char>(cv::Point(it->pt.x, it->pt.y)) == 0)
                    {
                        continue;
                    }


                    //consider those weak features
                    if (it->response < 20)
                    {
                        //variance of a patch
                        float textureness;

                        //check if this feature is too close to the image border
                        if (it->pt.x - 2 < 0 || it->pt.y - 2 < 0 ||
                            it->pt.x + 2 > im1.cols - 1 || it->pt.y + 2 > im1.rows - 1)
                            continue;

                        //patch around the feature
                        patch = im1(cv::Rect(it->pt.x - 2, it->pt.y - 2, 5, 5));

                        //computes variance
                        //Scalar mean, stddev;
                        //cv::meanStdDev(patch, mean, stddev);

                        ////finds textureless feature patch
                        //if(stddev.val[0] < VARIANCE)
                            //continue;
                    }

                    //runs to here means this feature will be kept
                    cv::rectangle(curMask,
                        //cv::Point(round(it->pt.x-MASK_RADIUS/2),round(it->pt.y-MASK_RADIUS/2)),//upperLeft
                        //cv::Point(round(it->pt.x+MASK_RADIUS/2),round(it->pt.y+MASK_RADIUS/2)),//downRight
                        cv::Point(int(it->pt.x - MASK_RADIUS / 2 + .5), int(it->pt.y - MASK_RADIUS / 2 + .5)),//upperLeft
                        cv::Point(int(it->pt.x + MASK_RADIUS / 2 + .5), int(it->pt.y + MASK_RADIUS / 2 + .5)),//downRight
                        cv::Scalar(0), -1);



                    allFeas.push_back(cv::Point2f(it->pt.x, it->pt.y));


                    ++numActiveTracks;

                    n++;
                }

                //recollects unused room
                if (room > n)
                {
                    // 		  unusedRoom += (room - n);
                }
                else
                {
                    //records hungry grid's index and how hungry they are
                    hungryGrid.push_back(std::make_pair(i, (end - it)));

                    //sums up to get the total hungry degree
                    hungry += hungryGrid.back().second;
                }
            }
        }


        //begin of second round
        unusedRoom = MaxTracks - numActiveTracks;

        //resampling for the second round
        if (unusedRoom > minToAdd)
        {
            vector<pair<int, float> >::iterator it = hungryGrid.begin(), end = hungryGrid.end();
            for (;it != end;it++)
            {

                //rowIndx for cells
                int celly = it->first / hgrids.x;

                //colIndx for cells
                int cellx = it->first - celly * hgrids.x;

                //rowRang for pixels
                Range row_range((celly*im1.rows) / hgrids.y, ((celly + 1)*im1.rows) / hgrids.y);

                //colRange for pixels
                Range col_range((cellx*im1.cols) / hgrids.x, ((cellx + 1)*im1.cols) / hgrids.x);

                //how much food can we give it
                room = floor((float)(unusedRoom * it->second) / (float)hungry);

                //add more features to this grid
                vector<KeyPoint>::iterator itPts = sub_keypoints[it->first].end() - (it->second),
                    endPts = sub_keypoints[it->first].end();
                for (int m = 0;
                    m < room && itPts != endPts;
                    itPts++)
                {
                    //transform grid based position to image based position
                    itPts->pt.x += col_range.start;
                    itPts->pt.y += row_range.start;

                    //check is features are being too close
                    if (curMask.at<unsigned char>(cv::Point(itPts->pt.x, itPts->pt.y)) == 0)
                    {
                        continue;
                    }

                    //consider those weak features
                    if (itPts->response < 20)
                    {
                        //variance of a patch
                        float textureness;

                        //check if this feature is too close to the image border
                        if (itPts->pt.x - 2 < 0 || itPts->pt.y - 2 < 0 ||
                            itPts->pt.x + 2 > im1.cols - 1 || itPts->pt.y + 2 > im1.rows - 1)
                            continue;

                        //patch around the feature
                        patch = im1(cv::Rect(itPts->pt.x - 2, itPts->pt.y - 2, 5, 5));

                        //computes variance
                        //Scalar mean, stddev;
                        //cv::meanStdDev(patch, mean, stddev);

                        //finds textureless feature patch
                        //if(stddev.val[0] < 4)
                            //continue;
                    }

                    //runs to here means this feature will be kept
                    cv::rectangle(curMask,
                        //cv::Point(round(itPts->pt.x-MASK_RADIUS/2),round(itPts->pt.y-MASK_RADIUS/2)),//upperLeft
                        //cv::Point(round(itPts->pt.x+MASK_RADIUS/2),round(itPts->pt.y+MASK_RADIUS/2)),//downRight
                        cv::Point(int(itPts->pt.x - MASK_RADIUS / 2 + .5), int(itPts->pt.y - MASK_RADIUS / 2 + .5)),//upperLeft
                        cv::Point(int(itPts->pt.x + MASK_RADIUS / 2 + .5), int(itPts->pt.y + MASK_RADIUS / 2 + .5)),//downRight
                        cv::Scalar(0), -1);

                    allFeas.push_back(cv::Point2f(itPts->pt.x, itPts->pt.y));

                    //counts the fatures that have added
                    m++;
                }
            }
        }
    }
    return 1;
}





