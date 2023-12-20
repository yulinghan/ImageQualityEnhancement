#include "orb_keypoint.hpp"

MyOrbKeyPointTest::MyOrbKeyPointTest() {
}

MyOrbKeyPointTest::~MyOrbKeyPointTest() {
}

Mat MyOrbKeyPointTest::CornersShow(Mat src, vector<KeyPoint> corners) {
    Mat dst = src.clone();

    for(int i=0; i<corners.size(); i++) {
        circle(dst, corners[i].pt, 3, Scalar(255, 0, 0), -1);  // 画半径为1的圆(画点）
    }

    return dst;
}

void MyOrbKeyPointTest::MakeOffsets(int pixel[25], int rowStride, int patternSize)
{
    static const int offsets[16][2] =
    {
        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    //填充16个圆形邻域像素
    int k = 0;
    for( ; k<patternSize; k++) {
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
    }
    //第16个像素和第一个像素连接起来，形成圆形闭环
    for( ; k<25; k++) {
        pixel[k] = pixel[k - patternSize];
    }
}

int MyOrbKeyPointTest::CornerScore(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 8, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N];
    for( k = 0; k < N; k++) {
        d[k] = (short)(v - ptr[pixel[k]]);
    }

    int a0 = threshold;
    for(k=0; k<16; k+=2) {
        int a = std::min((int)d[k+1], (int)d[k+2]);
        a = std::min(a, (int)d[k+3]);
        if( a <= a0 )
            continue;
        a = std::min(a, (int)d[k+4]);
        a = std::min(a, (int)d[k+5]);
        a = std::min(a, (int)d[k+6]);
        a = std::min(a, (int)d[k+7]);
        a = std::min(a, (int)d[k+8]);
        a0 = std::max(a0, std::min(a, (int)d[k]));
        a0 = std::max(a0, std::min(a, (int)d[k+9]));
    }

    int b0 = -a0;
    for(k=0; k<16; k+=2) {
        int b = std::max((int)d[k+1], (int)d[k+2]);
        b = std::max(b, (int)d[k+3]);
        b = std::max(b, (int)d[k+4]);
        b = std::max(b, (int)d[k+5]);
        if( b >= b0 )
            continue;
        b = std::max(b, (int)d[k+6]);
        b = std::max(b, (int)d[k+7]);
        b = std::max(b, (int)d[k+8]);

        b0 = std::min(b0, std::max(b, (int)d[k]));
        b0 = std::min(b0, std::max(b, (int)d[k+9]));
    }

    threshold = -b0 - 1;

    return threshold;
}

//nonmax_suppression:非最大抑制
void MyOrbKeyPointTest::OrbKeyPoint(Mat img, vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression) {
    int patternSize = 16;

    int K = patternSize/2;
    int N = patternSize + K + 1;
    int i, j, k, pixel[25];
    MakeOffsets(pixel, (int)img.step, patternSize);

    keypoints.clear();
    threshold = min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++) {
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);
    }
    uchar buf[3][img.cols];
    int cpbuf[3][img.cols+1];

    for (unsigned idx = 0; idx < 3; ++idx) {
        memset(buf[idx], 0, img.cols);
    }

    for(i = 3; i < img.rows-2; i++) {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3] + 1;
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3) {
            j = 3;
            for( ; j < img.cols - 3; j++, ptr++) {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;

                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];
                if(d == 0) {
                    continue;
                }

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];
                if(d == 0) {
                    continue;
                }

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if(d & 1) {
                    int vt = v - threshold, count = 0;
                    for( k = 0; k < N; k++) {
                        int x = ptr[pixel[k]];
                        if(x < vt) {
                            if( ++count > K) {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)CornerScore(ptr, pixel, threshold);
                                break;
                            }
                        } else {
                            count = 0;
                        }
                    }
                }

                if(d & 2) {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)CornerScore(ptr, pixel, threshold);
                                break;
                            }
                        } else {
                            count = 0;
                        }
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if(i == 3) {
            continue;
        }

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3] + 1; // cornerpos[-1] is used to store a value
        ncorners = cornerpos[-1];

        for(k = 0; k < ncorners; k++) {
            j = cornerpos[k];
            int score = prev[j];
            if(!nonmax_suppression ||
                    (score > prev[j+1] && score > prev[j-1] &&
                     score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                     score > curr[j-1] && score > curr[j] && score > curr[j+1]) ) {
                keypoints.push_back(KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
}

vector<KeyPoint> MyOrbKeyPointTest::Run(Mat src, int fastThreshold) {
    vector<KeyPoint> keypoints;
    OrbKeyPoint(src, keypoints, fastThreshold, true);

    return keypoints;
}
