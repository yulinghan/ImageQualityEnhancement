#include "akaze_key_points.hpp"

MyAkazeKeyPointsTest::MyAkazeKeyPointsTest() {
}

MyAkazeKeyPointsTest::~MyAkazeKeyPointsTest() {
}

bool MyAkazeKeyPointsTest::isExtremum(int r, int c, Mat t, Mat m, Mat b) {
    int layerBorder = 5;
    if (r<=layerBorder || r >= t.rows-layerBorder || c<=layerBorder || c>=t.cols-layerBorder) {
        return false;
    }

    // 检查中间层候选点是否超过阈值
    float candidate = m.at<float>(r, c);
    if (candidate < THRES) {
        return false; 
    }

    //如果在3x3x3内有任何响应值超过候选点的响应值，则候选点不是最大值
    for (int rr = -1; rr <=1; rr++){
        for (int cc = -1; cc <=1; cc++) {
            if ((t.at<float>(r+rr, c+cc)>=candidate)
                 || (((rr != 0 || cc != 0) && m.at<float>(r+rr, c+cc) >= candidate)) 
                 || (b.at<float>(r+rr, c+cc) >= candidate)) {
                return false;
            }
        }
    }

    return true;
}

//计算像素点的尺度及其x,y方向的偏导数
Mat MyAkazeKeyPointsTest::deriv3D(int r, int c, Mat t, Mat m, Mat b) {
    Mat dI = Mat::zeros(Size(1, 3), CV_32FC1);
    double dx, dy, ds;

    dx = (m.at<float>(r, c + 1) - m.at<float>(r, c - 1)) / 2.0;
    dy = (m.at<float>(r + 1, c) - m.at<float>(r - 1, c)) / 2.0;
    ds = (t.at<float>(r, c) - b.at<float>(r, c)) / 2.0;

    dI.at<float>(0, 0) = dx;
    dI.at<float>(0, 1) = dy;
    dI.at<float>(0, 2) = ds;

    return dI;
}

//计算像素点的三维Hessian矩阵
Mat MyAkazeKeyPointsTest::hessian3D(int r, int c, Mat t, Mat m, Mat b) {
    Mat H = Mat::zeros(Size(3, 3), CV_32FC1);
    double v, dxx, dyy, dss, dxy, dxs, dys;

    v = m.at<float>(r, c);
    dxx = m.at<float>(r, c + 1) + m.at<float>(r, c - 1) - 2 * v;
    dyy = m.at<float>(r + 1, c) + m.at<float>(r - 1, c) - 2 * v;
    dss = t.at<float>(r, c) + b.at<float>(r, c) - 2 * v;
    dxy = (m.at<float>(r + 1, c + 1) - m.at<float>(r + 1, c - 1) - 
            m.at<float>(r - 1, c + 1) + m.at<float>(r - 1, c - 1) ) / 4.0;
    dxs = (t.at<float>(r, c + 1) - t.at<float>(r, c - 1) - 
            b.at<float>(r, c + 1) + b.at<float>(r, c - 1) ) / 4.0;
    dys = ( t.at<float>(r + 1, c) - t.at<float>(r - 1, c) - 
            b.at<float>(r + 1, c) + b.at<float>(r - 1, c) ) / 4.0;

    H.at<float>(0, 0) = dxx;
    H.at<float>(0, 1) = dxy;
    H.at<float>(0, 2) = dxs;
    H.at<float>(1, 0) = dxy;
    H.at<float>(1, 1) = dyy;
    H.at<float>(1, 2) = dys;
    H.at<float>(2, 0) = dxs;
    H.at<float>(2, 1) = dys;
    H.at<float>(2, 2) = dss;

    return H;
}

//极值点插值函数
void MyAkazeKeyPointsTest::interpolateStep(int r, int c, Mat t, Mat m, Mat b, double &xi, double &xr, double &xc) {
    Mat H_inv, X;

    Mat dD = deriv3D( r, c, t, m, b );
    Mat H = hessian3D( r, c, t, m, b );
    invert(H, H_inv, CV_SVD);

    gemm(H_inv, dD, -1, Mat(), 0, X, 0);

    xi = X.at<float>(0, 2);
    xr = X.at<float>(0, 1);
    xc = X.at<float>(0, 0);
}

//通过插值来实现特征点的精确定位
bool MyAkazeKeyPointsTest::interpolateExtremum(int r, int c, Mat t, Mat m, Mat b, KeyPoint &key_point, int octave, int class_id){
    //通过插值，得到极值点位置的偏移量
    double xi = 0, xr = 0, xc = 0;
    interpolateStep(r, c, t, m, b, xi, xr, xc);

    //若偏移量都不超过0.5,则认为其足够接近实际的极值点
    if(fabs(xi)<0.5f && fabs(xr)<0.5f && fabs(xc)<0.5f) {
        key_point.pt.x = static_cast<float>((c + xc));
        key_point.pt.y = static_cast<float>((r + xr));
        key_point.octave = octave;
        key_point.class_id = class_id;
        return true;
    } else {
        return false;
    }
}
    
Mat MyAkazeKeyPointsTest::DispKeyPoint(Mat src, vector<KeyPoint> key_point_vec) {
    Mat dst = src.clone();
    
    for(int i=0; i<key_point_vec.size(); i++) {
        circle(dst, key_point_vec[i].pt*pow(2, key_point_vec[i].octave), 1, Scalar(255, 255, 255), -1);  // 画半径为1的圆(画点）
    }

    return dst;
}

vector<KeyPoint> MyAkazeKeyPointsTest::GetKeyPoints(vector<vector<Mat>> response_layer) {
    vector<KeyPoint> key_point_vec;

    Mat b, m, t;
    for (int o=0; o<response_layer.size(); o++) {
        for (int i=0; i<response_layer[i].size()-2; i++){
            b = response_layer[o][i];  //底层
            m = response_layer[o][i+1];//中间层
            t = response_layer[o][i+2];//顶层

            //遍历中间响应层，找到最大值所在的尺度和空间
            for(int r=0; r<t.rows; r++) {
                for(int c=0; c<t.cols; c++) {
                    if(isExtremum(r, c, t, m, b)) {
                        KeyPoint key_point;
                        if(interpolateExtremum(r, c, t, m, b, key_point, o, i+1)){
                            key_point_vec.push_back(key_point);
                        }
                    }
                }
            }
        }
    }
    return key_point_vec;
}

vector<vector<Mat>> MyAkazeKeyPointsTest::DeterminantHessianResponse(vector<vector<Mat>> scale_space_arr) {
    vector<vector<Mat>> response_arr2;                                                                                                                                                                     

    for(int i=0; i<scale_space_arr.size(); i++) {
        vector<Mat> response_arr;
        for(int j=0; j<scale_space_arr[i].size(); j++) {
            Mat Ixx, Ixy, Iyy, score;
            GetGradient(scale_space_arr[i][j], Ixx, Ixy, Iyy);
            score = ScoreImg(Ixx, Ixy, Iyy);
            response_arr.push_back(score);
        }
        response_arr2.push_back(response_arr);
    }                                                                                                                                                                                                      

    return response_arr2;
}

void MyAkazeKeyPointsTest::GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy){
    Mat sobelx = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat sobely = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    Mat Ix, Iy;
    Point point(-1, -1);
    filter2D(src, Ix, -1, sobelx, point, 0, BORDER_CONSTANT);
    filter2D(src, Iy, -1, sobely, point, 0, BORDER_CONSTANT);

    Ixx = Ix.mul(Ix);
    Iyy = Iy.mul(Iy);
    Ixy = Ix.mul(Iy);
}

Mat MyAkazeKeyPointsTest::ScoreImg(Mat Ixx, Mat Ixy, Mat Iyy) {
    Mat res = Mat::zeros(Ixx.size(),CV_32FC1);

    for(int r = 0; r < Ixx.rows; r++) {
        for(int c = 0; c < Ixx.cols; c++) {
            float score = (Ixx.at<float>(r,c) * Iyy.at<float>(r,c) - 0.81f * Ixy.at<float>(r,c) * Ixy.at<float>(r,c));;
            res.at<float>(r,c) = score;
        }
    }
    return res;
}

vector<KeyPoint> MyAkazeKeyPointsTest::run(vector<vector<Mat>> scale_space_arr) {
    vector<vector<Mat>> response_arr2 = DeterminantHessianResponse(scale_space_arr);
    vector<KeyPoint> key_point_vec = GetKeyPoints(response_arr2);

    return key_point_vec;
}
