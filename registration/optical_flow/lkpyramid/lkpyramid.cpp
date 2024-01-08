#include "lkpyramid.hpp"

LKPyramid::LKPyramid() {
}

LKPyramid::~LKPyramid() {
}

vector<KeyPoint> LKPyramid::MyCalcOpticalFlowPyrLK(vector<Mat> prev_pyr, vector<Mat> next_pyr, vector<KeyPoint> prev_points, 
                    vector<uchar> status, vector<float> error, Size winSize, int maxLevel, TermCriteria criteria) {
    vector<KeyPoint> next_points;
    next_points.resize(prev_points.size());
    Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
    int lvlStep = 2;
    Mat IWinBuf = Mat::zeros(winSize, CV_16SC1);
    Mat derivIWinBuf = Mat::zeros(winSize, CV_16SC2);

    for(int level=maxLevel; level>=0; level--){
        Mat derivI = prev_pyr[level*lvlStep+1];
        Mat I = prev_pyr[level*lvlStep];
        Mat J = next_pyr[level*lvlStep];

        for(int ptidx=0; ptidx<prev_points.size(); ptidx++) {
            Point2f prevPt = prev_points[ptidx].pt*(float)(1./(1 << level));
            Point2f nextPt;
            if(level == maxLevel){
                nextPt = prevPt;
            } else {
                nextPt = next_points[ptidx].pt*2.f;
            }
            next_points[ptidx].pt = nextPt;

            Point2i iprevPt, inextPt;
            prevPt -= halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            if(iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
                    iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows) {
                if(level == 0) {
                    status[ptidx] = false;
                    error[ptidx] = 0;
                }
                continue;
            }

            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;
            const int W_BITS = 14, W_BITS1 = 14;
            const float FLT_SCALE = 1.f/(1 << 20);
            int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            int dstep = (int)(derivI.step/derivI.elemSize1());
            int stepI = (int)(I.step/I.elemSize1());
            int stepJ = (int)(J.step/J.elemSize1());
            float iA11 = 0, iA12 = 0, iA22 = 0;
            float A11, A12, A22;

            // extract the patch from the first image, compute covariation matrix of derivatives
            int x, y;
            for(y = 0; y < winSize.height; y++) {
                const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x;
                const short* dsrc = derivI.ptr<short>() + (y + iprevPt.y)*dstep + iprevPt.x*2;

                short* Iptr = IWinBuf.ptr<short>(y);
                short* dIptr = derivIWinBuf.ptr<short>(y);

                for(x=0; x < winSize.width; x++, dsrc += 2, dIptr += 2) {
                    int ival = CV_DESCALE(src[x]*iw00 + src[x+1]*iw01 +
                            src[x+stepI]*iw10 + src[x+stepI+1]*iw11, W_BITS1-5);

                    int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[2]*iw01 +
                            dsrc[dstep]*iw10 + dsrc[dstep+3]*iw11, W_BITS1);

                    int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[2+1]*iw01 + dsrc[dstep+1]*iw10 +
                            dsrc[dstep+2+1]*iw11, W_BITS1);

                    Iptr[x]  = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;

                    iA11 += (float)(ixval*ixval);
                    iA12 += (float)(ixval*iyval);
                    iA22 += (float)(iyval*iyval);
                }
            }

            A11 = iA11*FLT_SCALE;
            A12 = iA12*FLT_SCALE;
            A22 = iA22*FLT_SCALE;

            float D = A11*A22 - A12*A12;
            float minEig = (A22 + A11 - sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winSize.width*winSize.height);

            error[ptidx] = (float)minEig;

            if(minEig < minEigThreshold_ || D < FLT_EPSILON) {
                if(level == 0) {
                    status[ptidx] = false;
                }
                continue;
            }

            D = 1.f/D;

            nextPt -= halfWin;
            Point2f prevDelta;

            for(int j = 0; j < criteria.maxCount; j++) {
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                if(inextPt.x < -winSize.width || inextPt.x >= J.cols ||
                        inextPt.y < -winSize.height || inextPt.y >= J.rows) {
                    if(level == 0) {
                        status[ptidx] = false;
                    }
                    break;
                }

                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                float ib1 = 0, ib2 = 0;
                float b1, b2;

                for(y = 0; y < winSize.height; y++) {
                    const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x;
                    const short* Iptr = IWinBuf.ptr<short>(y);
                    const short* dIptr = derivIWinBuf.ptr<short>(y);

                    for(x=0; x<winSize.width; x++, dIptr+=2) {
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+1]*iw01 +
                                Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+1]*iw11,
                                W_BITS1-5) - Iptr[x];
                        ib1 += (float)(diff*dIptr[0]);
                        ib2 += (float)(diff*dIptr[1]);
                    }
                }

                b1 = ib1*FLT_SCALE;
                b2 = ib2*FLT_SCALE;


                Point2f delta( (float)((A12*b2 - A22*b1) * D),
                        (float)((A12*b1 - A11*b2) * D));

                nextPt += delta;
                next_points[ptidx].pt = nextPt + halfWin;

                if(delta.ddot(delta) <= criteria.epsilon) {
                    break;
                }

                if(j>0 && abs(delta.x + prevDelta.x) < 0.01 &&
                        abs(delta.y + prevDelta.y) < 0.01) {
                    next_points[ptidx].pt -= delta*0.5f;
                    break;
                }
                prevDelta = delta;
            }
/*
            if( status[ptidx] && err && level == 0 && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0 )
            {
                Point2f nextPoint = nextPts[ptidx] - halfWin;
                Point inextPoint;

                inextPoint.x = cvFloor(nextPoint.x);
                inextPoint.y = cvFloor(nextPoint.y);

                if( inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
                        inextPoint.y < -winSize.height || inextPoint.y >= J.rows )
                {
                    if( status )
                        status[ptidx] = false;
                    continue;
                }

                float aa = nextPoint.x - inextPoint.x;
                float bb = nextPoint.y - inextPoint.y;
                iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
                iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
                iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                float errval = 0.f;

                for( y = 0; y < winSize.height; y++ )
                {
                    const uchar* Jptr = J.ptr() + (y + inextPoint.y)*stepJ + inextPoint.x*cn;
                    const short* Iptr = IWinBuf.ptr<short>(y);

                    for( x = 0; x < winSize.width*cn; x++ )
                    {
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                W_BITS1-5) - Iptr[x];
                        errval += std::abs((float)diff);
                    }
                }
                err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height);
            }
*/
        }
    }
    
    cout << "1111" << endl;

    return next_points;
}

vector<KeyPoint> LKPyramid::run(Mat src1, Mat src2, vector<KeyPoint> prev_points) {
    vector<KeyPoint> next_points;

    vector<uchar> status(prev_points.size(), 1);
    vector<float> error(prev_points.size(), -1);

    int TRACKING_HSIZE = 8;
    int LK_PYRAMID_LEVEL = 4;
    vector<Mat> prev_pyr, next_pyr;
    buildOpticalFlowPyramid(src1, prev_pyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);
    buildOpticalFlowPyramid(src2, next_pyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);

    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
/*
    vector<Point2f> allFeas, points1;
    for(int i=0; i<prev_points.size(); i++) {
        allFeas.push_back(prev_points[i].pt);
    }

    points1.resize(allFeas.size());
    calcOpticalFlowPyrLK(prev_pyr,
            next_pyr,
            cv::Mat(allFeas),
            cv::Mat(points1),
            cv::Mat(status),// '1' indicates successfull OF from points0
            cv::Mat(error),
            Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1),//size of searching window for each Pyramid level
            LK_PYRAMID_LEVEL,// now is 4, the maximum Pyramid levels
            criteria);

    cout << "!!!! points1.size():" << points1.size() << endl;

    next_points.resize(points1.size());
    for(int i=0; i<points1.size(); i++) {
        next_points[i].pt = points1[i];
    }
*/

    next_points = MyCalcOpticalFlowPyrLK(prev_pyr, next_pyr, prev_points, status, error, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, criteria);

    return next_points;
}
