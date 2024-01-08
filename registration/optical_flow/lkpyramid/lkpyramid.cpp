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
            float FLT_SCALE = 1.f/(1 << 20);
            float iw00 = (1.f - a)*(1.f - b);
            float iw01 = a*(1.f - b);
            float iw10 = (1.f - a)*b;
            float iw11 = 1.f - iw00 - iw01 - iw10;

            int dstep = (int)(derivI.step/derivI.elemSize1());
            int stepI = (int)(I.step/I.elemSize1());
            int stepJ = (int)(J.step/J.elemSize1());
            float A11=0, A12=0, A22=0;

            // extract the patch from the first image, compute covariation matrix of derivatives
            int x, y;
            for(y = 0; y < winSize.height; y++) {
                const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x;
                const short* dsrc = derivI.ptr<short>() + (y + iprevPt.y)*dstep + iprevPt.x*2;

                short* Iptr = IWinBuf.ptr<short>(y);
                short* dIptr = derivIWinBuf.ptr<short>(y);

                for(x=0; x < winSize.width; x++, dsrc += 2, dIptr += 2) {
                    int ival = src[x]*iw00 + src[x+1]*iw01 + src[x+stepI]*iw10 + src[x+stepI+1]*iw11;
                    int ixval = dsrc[0]*iw00 + dsrc[2]*iw01 + dsrc[dstep]*iw10 + dsrc[dstep+3]*iw11;
                    int iyval = dsrc[1]*iw00 + dsrc[2+1]*iw01 + dsrc[dstep+1]*iw10 + dsrc[dstep+2+1]*iw11;

                    Iptr[x]  = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;

                    A11 += (float)(ixval*ixval);
                    A12 += (float)(ixval*iyval);
                    A22 += (float)(iyval*iyval);
                }
            }

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
                float iw00 = (1.f - a)*(1.f - b);
                float iw01 = a*(1.f - b);
                float iw10 = (1.f - a)*b;
                float iw11 = 1.f - iw00 - iw01 - iw10;
                float b1=0, b2=0;

                for(y = 0; y < winSize.height; y++) {
                    const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x;
                    const short* Iptr = IWinBuf.ptr<short>(y);
                    const short* dIptr = derivIWinBuf.ptr<short>(y);

                    for(x=0; x<winSize.width; x++, dIptr+=2) {
                        int diff = (Jptr[x]*iw00 + Jptr[x+1]*iw01 + Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+1]*iw11) - Iptr[x];
                        b1 += (float)(diff*dIptr[0]);
                        b2 += (float)(diff*dIptr[1]);
                    }
                }

                Point2f delta((float)((A12*b2 - A22*b1) * D), (float)((A12*b1 - A11*b2) * D));

                nextPt += delta;
                next_points[ptidx].pt = nextPt + halfWin;

                if(delta.ddot(delta) <= criteria.epsilon) {
                    break;
                }

                if(j>0 && abs(delta.x + prevDelta.x) < 0.01 && abs(delta.y + prevDelta.y) < 0.01) {
                    next_points[ptidx].pt -= delta*0.5f;
                    break;
                }
                prevDelta = delta;
            }
        }
    }
    
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
    next_points = MyCalcOpticalFlowPyrLK(prev_pyr, next_pyr, prev_points, status, error, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, criteria);

    return next_points;
}
