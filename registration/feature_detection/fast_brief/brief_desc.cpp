#include "brief_desc.hpp"

MyBriefTest::MyBriefTest() {
}

MyBriefTest::~MyBriefTest() {
}

int MyBriefTest::smoothedSum(Mat sum, KeyPoint pt, int y, int x) {
    int HALF_KERNEL = KERNEL_SIZE / 2;

    int img_y = (int)(pt.pt.y + 0.5) + y;
    int img_x = (int)(pt.pt.x + 0.5) + x;

    return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
        - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
        - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
        + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}

void MyBriefTest::runByImageBorder(vector<KeyPoint>& keypoints, Size imageSize, int borderSize) {
    struct RoiPredicate {
        RoiPredicate( const Rect& _r ) : r(_r)
        {}

        bool operator()( const KeyPoint& keyPt ) const {
            return !r.contains( keyPt.pt );
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

Mat MyBriefTest::pixelTests16(Mat sum, vector<KeyPoint> corners) {
    Mat descriptors = Mat::zeros(Size(16, corners.size()), CV_8UC1);

    for(int i=0; i<(int)corners.size(); i++) {
#define SMOOTHED(y, x) smoothedSum(sum, corners[i], y, x)
        uchar* desc = descriptors.ptr(i);

        desc[0]  = (uchar)(((SMOOTHED(-2, -1) < SMOOTHED(7, -1)) << 7) + ((SMOOTHED(-14, -1) < SMOOTHED(-3, 3)) << 6)
                    + ((SMOOTHED(1, -2) < SMOOTHED(11, 2)) << 5) + ((SMOOTHED(1, 6) < SMOOTHED(-10, -7)) << 4)
                    + ((SMOOTHED(13, 2) < SMOOTHED(-1, 0)) << 3) + ((SMOOTHED(-14, 5) < SMOOTHED(5, -3)) << 2)
                    + ((SMOOTHED(-2, 8) < SMOOTHED(2, 4)) << 1) + ((SMOOTHED(-11, 8) < SMOOTHED(-15, 5)) << 0));

        desc[1]  = (uchar)(((SMOOTHED(-6, -23) < SMOOTHED(8, -9)) << 7) + ((SMOOTHED(-12, 6) < SMOOTHED(-10, 8)) << 6)
                    + ((SMOOTHED(-3, -1) < SMOOTHED(8, 1)) << 5) + ((SMOOTHED(3, 6) < SMOOTHED(5, 6)) << 4)
                    + ((SMOOTHED(-7, -6) < SMOOTHED(5, -5)) << 3) + ((SMOOTHED(22, -2) < SMOOTHED(-11, -8)) << 2)
                    + ((SMOOTHED(14, 7) < SMOOTHED(8, 5)) << 1) + ((SMOOTHED(-1, 14) < SMOOTHED(-5, -14)) << 0));

        desc[2]  = (uchar)(((SMOOTHED(-14, 9) < SMOOTHED(2, 0)) << 7) + ((SMOOTHED(7, -3) < SMOOTHED(22, 6)) << 6) 
                    + ((SMOOTHED(-6, 6) < SMOOTHED(-8, -5)) << 5) + ((SMOOTHED(-5, 9) < SMOOTHED(7, -1)) << 4) 
                    + ((SMOOTHED(-3, -7) < SMOOTHED(-10, -18)) << 3) + ((SMOOTHED(4, -5) < SMOOTHED(0, 11)) << 2) 
                    + ((SMOOTHED(2, 3) < SMOOTHED(9, 10)) << 1) + ((SMOOTHED(-10, 3) < SMOOTHED(4, 9)) << 0));

        desc[3]  = (uchar)(((SMOOTHED(0, 12) < SMOOTHED(-3, 19)) << 7) + ((SMOOTHED(1, 15) < SMOOTHED(-11, -5)) << 6) 
                    + ((SMOOTHED(14, -1) < SMOOTHED(7, 8)) << 5) + ((SMOOTHED(7, -23) < SMOOTHED(-5, 5)) << 4) 
                    + ((SMOOTHED(0, -6) < SMOOTHED(-10, 17)) << 3) + ((SMOOTHED(13, -4) < SMOOTHED(-3, -4)) << 2) 
                    + ((SMOOTHED(-12, 1) < SMOOTHED(-12, 2)) << 1) + ((SMOOTHED(0, 8) < SMOOTHED(3, 22)) << 0));

        desc[4]  = (uchar)(((SMOOTHED(-13, 13) < SMOOTHED(3, -1)) << 7) + ((SMOOTHED(-16, 17) < SMOOTHED(6, 10)) << 6) 
                    + ((SMOOTHED(7, 15) < SMOOTHED(-5, 0)) << 5) + ((SMOOTHED(2, -12) < SMOOTHED(19, -2)) << 4) 
                    + ((SMOOTHED(3, -6) < SMOOTHED(-4, -15)) << 3) + ((SMOOTHED(8, 3) < SMOOTHED(0, 14)) << 2) 
                    + ((SMOOTHED(4, -11) < SMOOTHED(5, 5)) << 1) + ((SMOOTHED(11, -7) < SMOOTHED(7, 1)) << 0));

        desc[5]  = (uchar)(((SMOOTHED(6, 12) < SMOOTHED(21, 3)) << 7) + ((SMOOTHED(-3, 2) < SMOOTHED(14, 1)) << 6) 
                    + ((SMOOTHED(5, 1) < SMOOTHED(-5, 11)) << 5) + ((SMOOTHED(3, -17) < SMOOTHED(-6, 2)) << 4) 
                    + ((SMOOTHED(6, 8) < SMOOTHED(5, -10)) << 3) + ((SMOOTHED(-14, -2) < SMOOTHED(0, 4)) << 2) 
                    + ((SMOOTHED(5, -7) < SMOOTHED(-6, 5)) << 1) + ((SMOOTHED(10, 4) < SMOOTHED(4, -7)) << 0));

        desc[6]  = (uchar)(((SMOOTHED(22, 0) < SMOOTHED(7, -18)) << 7) + ((SMOOTHED(-1, -3) < SMOOTHED(0, 18)) << 6) 
                    + ((SMOOTHED(-4, 22) < SMOOTHED(-5, 3)) << 5) + ((SMOOTHED(1, -7) < SMOOTHED(2, -3)) << 4) 
                    + ((SMOOTHED(19, -20) < SMOOTHED(17, -2)) << 3) + ((SMOOTHED(3, -10) < SMOOTHED(-8, 24)) << 2) 
                    + ((SMOOTHED(-5, -14) < SMOOTHED(7, 5)) << 1) + ((SMOOTHED(-2, 12) < SMOOTHED(-4, -15)) << 0));

        desc[7]  = (uchar)(((SMOOTHED(4, 12) < SMOOTHED(0, -19)) << 7) + ((SMOOTHED(20, 13) < SMOOTHED(3, 5)) << 6) 
                    + ((SMOOTHED(-8, -12) < SMOOTHED(5, 0)) << 5) + ((SMOOTHED(-5, 6) < SMOOTHED(-7, -11)) << 4) 
                    + ((SMOOTHED(6, -11) < SMOOTHED(-3, -22)) << 3) + ((SMOOTHED(15, 4) < SMOOTHED(10, 1)) << 2) 
                    + ((SMOOTHED(-7, -4) < SMOOTHED(15, -6)) << 1) + ((SMOOTHED(5, 10) < SMOOTHED(0, 24)) << 0));

        desc[8]  = (uchar)(((SMOOTHED(3, 6) < SMOOTHED(22, -2)) << 7) + ((SMOOTHED(-13, 14) < SMOOTHED(4, -4)) << 6) 
                    + ((SMOOTHED(-13, 8) < SMOOTHED(-18, -22)) << 5) + ((SMOOTHED(-1, -1) < SMOOTHED(-7, 3)) << 4) 
                    + ((SMOOTHED(-19, -12) < SMOOTHED(4, 3)) << 3) + ((SMOOTHED(8, 10) < SMOOTHED(13, -2)) << 2) 
                    + ((SMOOTHED(-6, -1) < SMOOTHED(-6, -5)) << 1) + ((SMOOTHED(2, -21) < SMOOTHED(-3, 2)) << 0));

        desc[9]  = (uchar)(((SMOOTHED(4, -7) < SMOOTHED(0, 16)) << 7) + ((SMOOTHED(-6, -5) < SMOOTHED(-12, -1)) << 6) 
                    + ((SMOOTHED(1, -1) < SMOOTHED(9, 18)) << 5) + ((SMOOTHED(-7, 10) < SMOOTHED(-11, 6)) << 4) 
                    + ((SMOOTHED(4, 3) < SMOOTHED(19, -7)) << 3) + ((SMOOTHED(-18, 5) < SMOOTHED(-4, 5)) << 2) 
                    + ((SMOOTHED(4, 0) < SMOOTHED(-20, 4)) << 1) + ((SMOOTHED(7, -11) < SMOOTHED(18, 12)) << 0));

        desc[10] = (uchar)(((SMOOTHED(-20, 17) < SMOOTHED(-18, 7)) << 7) + ((SMOOTHED(2, 15) < SMOOTHED(19, -11)) << 6) 
                    + ((SMOOTHED(-18, 6) < SMOOTHED(-7, 3)) << 5) + ((SMOOTHED(-4, 1) < SMOOTHED(-14, 13)) << 4) 
                    + ((SMOOTHED(17, 3) < SMOOTHED(2, -8)) << 3) + ((SMOOTHED(-7, 2) < SMOOTHED(1, 6)) << 2) 
                    + ((SMOOTHED(17, -9) < SMOOTHED(-2, 8)) << 1) + ((SMOOTHED(-8, -6) < SMOOTHED(-1, 12)) << 0));

        desc[11] = (uchar)(((SMOOTHED(-2, 4) < SMOOTHED(-1, 6)) << 7) + ((SMOOTHED(-2, 7) < SMOOTHED(6, 8)) << 6) 
                    + ((SMOOTHED(-8, -1) < SMOOTHED(-7, -9)) << 5) + ((SMOOTHED(8, -9) < SMOOTHED(15, 0)) << 4) 
                    + ((SMOOTHED(0, 22) < SMOOTHED(-4, -15)) << 3) + ((SMOOTHED(-14, -1) < SMOOTHED(3, -2)) << 2) 
                    + ((SMOOTHED(-7, -4) < SMOOTHED(17, -7)) << 1) + ((SMOOTHED(-8, -2) < SMOOTHED(9, -4)) << 0));

        desc[12] = (uchar)(((SMOOTHED(5, -7) < SMOOTHED(7, 7)) << 7) + ((SMOOTHED(-5, 13) < SMOOTHED(-8, 11)) << 6) 
                    + ((SMOOTHED(11, -4) < SMOOTHED(0, 8)) << 5) + ((SMOOTHED(5, -11) < SMOOTHED(-9, -6)) << 4) 
                    + ((SMOOTHED(2, -6) < SMOOTHED(3, -20)) << 3) + ((SMOOTHED(-6, 2) < SMOOTHED(6, 10)) << 2) 
                    + ((SMOOTHED(-6, -6) < SMOOTHED(-15, 7)) << 1) + ((SMOOTHED(-6, -3) < SMOOTHED(2, 1)) << 0));

        desc[13] = (uchar)(((SMOOTHED(11, 0) < SMOOTHED(-3, 2)) << 7) + ((SMOOTHED(7, -12) < SMOOTHED(14, 5)) << 6) 
                    + ((SMOOTHED(0, -7) < SMOOTHED(-1, -1)) << 5) + ((SMOOTHED(-16, 0) < SMOOTHED(6, 8)) << 4) 
                    + ((SMOOTHED(22, 11) < SMOOTHED(0, -3)) << 3) + ((SMOOTHED(19, 0) < SMOOTHED(5, -17)) << 2) 
                    + ((SMOOTHED(-23, -14) < SMOOTHED(-13, -19)) << 1) + ((SMOOTHED(-8, 10) < SMOOTHED(-11, -2)) << 0));

        desc[14] = (uchar)(((SMOOTHED(-11, 6) < SMOOTHED(-10, 13)) << 7) + ((SMOOTHED(1, -7) < SMOOTHED(14, 0)) << 6) 
                    + ((SMOOTHED(-12, 1) < SMOOTHED(-5, -5)) << 5) + ((SMOOTHED(4, 7) < SMOOTHED(8, -1)) << 4) 
                    + ((SMOOTHED(-1, -5) < SMOOTHED(15, 2)) << 3) + ((SMOOTHED(-3, -1) < SMOOTHED(7, -10)) << 2) 
                    + ((SMOOTHED(3, -6) < SMOOTHED(10, -18)) << 1) + ((SMOOTHED(-7, -13) < SMOOTHED(-13, 10)) << 0));

        desc[15] = (uchar)(((SMOOTHED(1, -1) < SMOOTHED(13, -10)) << 7) + ((SMOOTHED(-19, 14) < SMOOTHED(8, -14)) << 6) 
                    + ((SMOOTHED(-4, -13) < SMOOTHED(7, 1)) << 5) + ((SMOOTHED(1, -2) < SMOOTHED(12, -7)) << 4) 
                    + ((SMOOTHED(3, -5) < SMOOTHED(1, -5)) << 3) + ((SMOOTHED(-2, -2) < SMOOTHED(8, -10)) << 2) 
                    + ((SMOOTHED(2, 14) < SMOOTHED(8, 7)) << 1) + ((SMOOTHED(3, 9) < SMOOTHED(8, 2)) << 0));
    }

    return descriptors;
}

Mat MyBriefTest::Run(Mat src, vector<KeyPoint> corners) {
    //生成图像积分图(平滑噪声时候，这里使用均值滤波，代替论文上的高斯滤波)
    Mat sum;
    integral(src, sum, CV_32S);
  
    //边界区域上的特征点直接丢弃
    runByImageBorder(corners, src.size(), PATCH_SIZE/2 + KERNEL_SIZE/2);

    //生存16位特征描述
    //注意opencv实现上，把随机取点的随机数直接固化了
    //最终每个特征点使用16个uchar数据保存特征数据
    Mat descriptors = pixelTests16(sum, corners);

    return descriptors;
}
