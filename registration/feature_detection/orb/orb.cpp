#include "orb.hpp"
#include "orb_desc.hpp"
#include "orb_keypoint.hpp"

MyOrbTest::MyOrbTest() {

}

MyOrbTest::~MyOrbTest() {

}

void MyOrbTest::runByImageBorder(vector<KeyPoint>& keypoints, Size imageSize, int borderSize) {
    struct RoiPredicate {
        RoiPredicate(const Rect& _r) : r(_r)
        {}

        bool operator()(const KeyPoint& keyPt) const {
            return !r.contains(keyPt.pt);
        }
        Rect r;
    };

    if(borderSize > 0) {
        if(imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2) {
            keypoints.clear();
        } else {
            keypoints.erase(remove_if(keypoints.begin(), keypoints.end(),
                        RoiPredicate(Rect(Point(borderSize, borderSize), Point(imageSize.width - borderSize, imageSize.height - borderSize)))),
                    keypoints.end() );
        }
    }
}

struct KeypointResponseGreater {
    inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const {
        return kp1.response > kp2.response;
    }
};

void MyOrbTest::HarrisResponses(Mat src, vector<KeyPoint> &key_points, int blockSize, float HARRIS_K) {
    for(int i=0; i<(int)key_points.size(); i++) {
        float Lxx = 0, Lyy = 0, Lxy = 0;
        for(int y = key_points[i].pt.y - blockSize; y <= key_points[i].pt.y + blockSize; y ++) {
            for(int x = key_points[i].pt.x - blockSize; x <= key_points[i].pt.x + blockSize; x ++) {
                float Lx = src.at<uchar>(y, x+1) - src.at<uchar>(y, x-1);
                float Ly = src.at<uchar>(y+1, x) - src.at<uchar>(y-1, x);
                Lxx += Lx*Lx; Lyy += Ly*Ly; Lxy += Lx*Ly;
            }
        }
        key_points[i].response =  (Lxx*Lyy - Lxy*Lxy) - HARRIS_K*(Lxx + Lyy)*(Lxx + Lyy);
    }
}
    
vector<KeyPoint> MyOrbTest::DistanceChoice(vector<KeyPoint> allKeypoints, float minDistance) {
    vector<KeyPoint> choice_key_points;

    for(int i=0; i<allKeypoints.size(); i++) {
        int cur_x = allKeypoints[i].pt.x * pow(2, allKeypoints[i].octave);
        int cur_y = allKeypoints[i].pt.y * pow(2, allKeypoints[i].octave);
        int x1 = cur_x - minDistance;
        int x2 = cur_x + minDistance;
        int y1 = cur_y - minDistance;
        int y2 = cur_y + minDistance;
        bool choice = true;
        for(int j=0; j<choice_key_points.size(); j++) {
            int pre_x = choice_key_points[j].pt.x * pow(2, choice_key_points[j].octave);
            int pre_y = choice_key_points[j].pt.y * pow(2, choice_key_points[j].octave);
            if((x1<pre_x) && (x2>pre_x) && (y1<pre_y) && (y2>pre_y)) {
                choice = false;
                break;
            }
        }
        if(choice) {
            choice_key_points.push_back(allKeypoints[i]);
        }
    }

    return choice_key_points;
}

void MyOrbTest::ICAngles(vector<Mat> src_pyr, vector<KeyPoint> &pts, int patchSize, int border) {
    int halfPatchSize = patchSize / 2;
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for(v = 0; v <= vmax; ++v) {
        umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));
    }
    // Make sure we are symmetric
    for(v = halfPatchSize, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    size_t ptidx, ptsize = pts.size();

    for(ptidx = 0; ptidx < ptsize; ptidx++) {
        int step = (int)src_pyr[pts[ptidx].octave].step1();
        uchar* center = &src_pyr[pts[ptidx].octave].at<uchar>(cvRound(pts[ptidx].pt.y) + border, cvRound(pts[ptidx].pt.x) + border);
        int m_01 = 0, m_10 = 0;

        // Treat the center line differently, v=0
        for(int u = -halfPatchSize; u <= halfPatchSize; ++u)
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for(int v = 1; v <= halfPatchSize; ++v) {
            // Proceed over the two lines
            int v_sum = 0;
            int d = umax[v];
            for(int u = -d; u <= d; ++u) {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }
        pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
    }
}

void MyOrbTest::computeKeyPoints(vector<Mat> src_pyr, vector<KeyPoint> &allKeypoints, int nfeatures, 
                    int edgeThreshold, int patchSize, int fastThreshold, int border) {

    for(int i=0; i<(int)src_pyr.size(); i++) {
        MyOrbKeyPointTest *my_orb_point_test = new MyOrbKeyPointTest();
        vector<KeyPoint> corners = my_orb_point_test->Run(src_pyr[i], fastThreshold);
        runByImageBorder(corners, src_pyr[i].size(), edgeThreshold);
        sort(corners.begin(), corners.end(), KeypointResponseGreater());

        int cur_features = nfeatures / pow(2, i);
        cur_features = min(cur_features, (int)corners.size());
        corners.erase(corners.begin()+cur_features, corners.end());

        float HARRIS_K = 0.04;
        HarrisResponses(src_pyr[i], corners, 7, HARRIS_K);

        for(int m=0; m<cur_features; m++) {
            corners[m].octave = i;
            corners[m].size = patchSize * pow(2, i);
            corners[m].pt.x = corners[m].pt.x - border;
            corners[m].pt.y = corners[m].pt.y - border;
            allKeypoints.push_back(corners[m]);
        }
    }

    sort(allKeypoints.begin(), allKeypoints.end(), KeypointResponseGreater());
    vector<KeyPoint> new_key_points = DistanceChoice(allKeypoints, 7);
    if(new_key_points.size()>nfeatures/2) {
        new_key_points.erase(new_key_points.begin()+nfeatures/2, new_key_points.end());
    }

    ICAngles(src_pyr, new_key_points, patchSize, border);
    for(int i=0; i<new_key_points.size(); i++) {
        new_key_points[i].pt = new_key_points[i].pt*pow(2, new_key_points[i].octave);
    }

    allKeypoints = new_key_points;
}

Mat MyOrbTest::CornersShow(Mat src, vector<KeyPoint> corners) {
    Mat dst = src.clone();

    for(int i=0; i<corners.size(); i++) {
        circle(dst, corners[i].pt, 3, Scalar(255, 0, 0), -1);  // 画半径为1的圆(画点）
//        cout << "keypoint[" << i << "]:" << corners[i].pt << ", angle:" << corners[i].angle << endl;
    }

    return dst;
}

void MyOrbTest::Run(Mat src) {
    int nlevels = 8;
    int nfeatures = 250;
    int HARRIS_BLOCK_SIZE = 9;
    int edgeThreshold = 31;
    int patchSize = 31;
    int fastThreshold = 20;
    int halfPatchSize = patchSize / 2;
    int descPatchSize = cvCeil(halfPatchSize*sqrt(2.0));
    int border = max(edgeThreshold, max(descPatchSize, HARRIS_BLOCK_SIZE/2))+1;

    vector<Mat> src_pyr;
    Mat cur_img = src;
    for(int level = 0; level < nlevels; level++) {
        Mat next_img;
        if(level > 0) {
            resize(cur_img, next_img, cur_img.size()/2);
        } else {
            next_img = cur_img.clone();
        }
        Mat next_border_img;
        copyMakeBorder(next_img, next_border_img, border, border, border, border,
                           BORDER_REFLECT_101+BORDER_ISOLATED);
        src_pyr.push_back(next_border_img);
        cur_img = next_img;
    }

    vector<KeyPoint> allKeypoints;
    computeKeyPoints(src_pyr, allKeypoints, nfeatures, edgeThreshold, patchSize, fastThreshold, border);

    Mat key_show = CornersShow(src, allKeypoints);
    imshow("key_show", key_show);

    MyOrbDescTest *my_orb_desc_test = new MyOrbDescTest();
    vector<vector<int>> descriptors = my_orb_desc_test->Run(src_pyr, allKeypoints, border);

    cout << "descriptors:" << descriptors.size() << ", descriptors[0]:" << descriptors[0].size() << endl;
}
