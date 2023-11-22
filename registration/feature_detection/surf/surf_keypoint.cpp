#include "surf_keypoint.hpp"

MySurfKeyPointTest::MySurfKeyPointTest() {
}

MySurfKeyPointTest::~MySurfKeyPointTest() {
}

// 每组盒子滤波器尺寸大小（默认5组4层）
// Oct1: 9,  15, 21, 27
// Oct2: 15, 27, 39, 51
// Oct3: 27, 51, 75, 99
// Oct4: 51, 99, 147,195
// Oct5: 99, 195,291,387
vector<vector<ResponseLayer>> MySurfKeyPointTest::buildResponseMap(Mat integ_mat) {
    vector<vector<ResponseLayer>> response_layer_arr;

    int filter_arr[5][4] = {{9,  15,  21,  27},
                            {15, 27,  39,  51},
                            {27, 51,  75,  99},
                            {51, 99,  147, 195},
                            {99, 195, 291, 387}};

    for(int i=0; i<OCTAVES; i++) {
        vector<ResponseLayer> response_layer_vec;
        for(int j=0; j<INTERVALS; j++) {
            ResponseLayer response_layer;

            response_layer.width     = integ_mat.cols / (INIT_SAMPLE*pow(2, i+1));
            response_layer.height    = integ_mat.rows / (INIT_SAMPLE*pow(2, i+1));
            response_layer.step      = INIT_SAMPLE*pow(2, i+1);
            response_layer.filter    = filter_arr[i][j];
            response_layer.responses = Mat::zeros(Size(response_layer.width, response_layer.height), CV_32FC1);
            response_layer.laplacian = Mat::zeros(Size(response_layer.width, response_layer.height), CV_32FC1);

            response_layer_vec.push_back(response_layer);
        }
        response_layer_arr.push_back(response_layer_vec);
    }

    for(int i=0; i<response_layer_arr.size(); i++) {
        for(int j=0; j<response_layer_arr[i].size(); j++) {
            int step = response_layer_arr[i][j].step;                      // 滤波器步长
            int b = (response_layer_arr[i][j].filter - 1) / 2;             // 滤波器边界
            int l = response_layer_arr[i][j].filter / 3;                   // 子滤波器(滤波器尺寸/3)
            int w = response_layer_arr[i][j].filter;                       // 滤波器大小
            float inverse_area = 1.f/(w*w);                             // 归一化因子
            float Dxx, Dyy, Dxy;

            int index = 0;
            for(int ar=0; ar<response_layer_arr[i][j].height; ar++) {
                for(int ac=0; ac<response_layer_arr[i][j].width; ac++) {
                    int r = ar * step;
                    int c = ac * step; 

                    Dxx = BoxIntegral(integ_mat, r - l + 1, c - b, 2*l - 1, w)
                        - BoxIntegral(integ_mat, r - l + 1, c - l / 2, 2*l - 1, l)*3;

                    Dyy = BoxIntegral(integ_mat, r - b, c - l + 1, w, 2*l - 1)
                        - BoxIntegral(integ_mat, r - l / 2, c - l + 1, l, 2*l - 1)*3;

                    Dxy = + BoxIntegral(integ_mat, r - l, c + 1, l, l)
                        + BoxIntegral(integ_mat, r + 1, c - l, l, l)
                        - BoxIntegral(integ_mat, r - l, c - l, l, l)
                        - BoxIntegral(integ_mat, r + 1, c + 1, l, l);

                    Dxx *= inverse_area;
                    Dyy *= inverse_area;
                    Dxy *= inverse_area;

                    response_layer_arr[i][j].responses.at<float>(ar, ac) = (Dxx * Dyy - 0.81f * Dxy * Dxy);
                    response_layer_arr[i][j].laplacian.at<float>(ar, ac)  = (Dxx + Dyy >= 0 ? 1 : 0);
                }
            }
        }
    }

    return response_layer_arr;
}

bool MySurfKeyPointTest::isExtremum(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b) {
    // 边界检测
    int layerBorder = (t.filter+1) / (2*t.step);
    if (r<=layerBorder || r >= t.height-layerBorder || c<=layerBorder || c>=t.width-layerBorder) {
        return false;
    }
    // 检查中间层候选点是否超过阈值
    float candidate = m.responses.at<float>(r, c);
    if (candidate < THRES) {
        return false; 
    }

    //如果在3x3x3内有任何响应值超过候选点的响应值，则候选点不是最大值
    for (int rr = -1; rr <=1; rr++){
        for (int cc = -1; cc <=1; cc++) {
            if ((t.responses.at<float>(r+rr, c+cc)>=candidate)
                 || (((rr != 0 || cc != 0) && m.responses.at<float>(r+rr, c+cc) >= candidate)) 
                 || (b.responses.at<float>(r+rr, c+cc) >= candidate)) {
                return false;
            }
        }
    }

    return true;
}

//计算像素点的尺度及其x,y方向的偏导数
Mat MySurfKeyPointTest::deriv3D(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b) {
    Mat dI = Mat::zeros(Size(1, 3), CV_32FC1);
    double dx, dy, ds;

    dx = (m.responses.at<float>(r, c + 1) - m.responses.at<float>(r, c - 1)) / 2.0;
    dy = (m.responses.at<float>(r + 1, c) - m.responses.at<float>(r - 1, c)) / 2.0;
    ds = (t.responses.at<float>(r, c) - b.responses.at<float>(r, c)) / 2.0;

    dI.at<float>(0, 0) = dx;
    dI.at<float>(0, 1) = dy;
    dI.at<float>(0, 2) = ds;

    return dI;
}

//计算像素点的三维Hessian矩阵
Mat MySurfKeyPointTest::hessian3D(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b) {
    Mat H = Mat::zeros(Size(3, 3), CV_32FC1);
    double v, dxx, dyy, dss, dxy, dxs, dys;

    v = m.responses.at<float>(r, c);
    dxx = m.responses.at<float>(r, c + 1) + m.responses.at<float>(r, c - 1) - 2 * v;
    dyy = m.responses.at<float>(r + 1, c) + m.responses.at<float>(r - 1, c) - 2 * v;
    dss = t.responses.at<float>(r, c) + b.responses.at<float>(r, c) - 2 * v;
    dxy = (m.responses.at<float>(r + 1, c + 1) - m.responses.at<float>(r + 1, c - 1) - 
            m.responses.at<float>(r - 1, c + 1) + m.responses.at<float>(r - 1, c - 1) ) / 4.0;
    dxs = (t.responses.at<float>(r, c + 1) - t.responses.at<float>(r, c - 1) - 
            b.responses.at<float>(r, c + 1) + b.responses.at<float>(r, c - 1) ) / 4.0;
    dys = ( t.responses.at<float>(r + 1, c) - t.responses.at<float>(r - 1, c) - 
            b.responses.at<float>(r + 1, c) + b.responses.at<float>(r - 1, c) ) / 4.0;

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
void MySurfKeyPointTest::interpolateStep(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b, 
        double &xi, double &xr, double &xc) {
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
bool MySurfKeyPointTest::interpolateExtremum(int r, int c, ResponseLayer t, ResponseLayer m, ResponseLayer b, MyKeyPoint &key_point){
    int filterStep = (m.filter - b.filter);

    //通过插值，得到极值点位置的偏移量
    double xi = 0, xr = 0, xc = 0;
    interpolateStep(r, c, t, m, b, xi, xr, xc);

    //若偏移量都不超过0.5,则认为其足够接近实际的极值点
    if(fabs(xi)<0.5f && fabs(xr)<0.5f && fabs(xc)<0.5f) {
        key_point.x = static_cast<float>((c + xc) * t.step);
        key_point.y = static_cast<float>((r + xr) * t.step);
        key_point.scale = static_cast<float>((0.1333f) * (m.filter + xi*filterStep));//0.1333f=1.2/9
        key_point.laplacian = m.laplacian.at<float>(r, c);

        return true;
    } else {
        return false;
    }
}
    
Mat MySurfKeyPointTest::DispKeyPoint(Mat src, vector<MyKeyPoint> key_point_vec) {
    Mat dst = src.clone();
    
    for(int i=0; i<key_point_vec.size(); i++) {
        Point p(key_point_vec[i].x, key_point_vec[i].y);//初始化点坐标为(20,20)
        circle(dst, p, 1, Scalar(255, 255, 255), -1);  // 画半径为1的圆(画点）
    }

    return dst;
}

vector<MyKeyPoint> MySurfKeyPointTest::GetKeyPoints(vector<vector<ResponseLayer>> response_layer) {
    vector<MyKeyPoint> key_point_vec;

    ResponseLayer b, m, t;
    for (int o=0; o<OCTAVES; o++) {
        for (int i=0; i<INTERVALS-2; i++){
            b = response_layer[o][i];  //底层
            m = response_layer[o][i+1];//中间层
            t = response_layer[o][i+2];//顶层

            //遍历中间响应层，找到最大值所在的尺度和空间
            for(int r=0; r<t.height; r++) {
                for(int c=0; c<t.width; c++) {
                    if(isExtremum(r, c, t, m, b)) {
                        MyKeyPoint key_point;
                        if(interpolateExtremum(r, c, t, m, b, key_point)){
                            key_point_vec.push_back(key_point);
                        }
                    }
                }
            }
        }
    }
    return key_point_vec;
}

vector<MyKeyPoint> MySurfKeyPointTest::run(Mat integ_mat) {
    vector<vector<ResponseLayer>> response_layer = buildResponseMap(integ_mat);
    vector<MyKeyPoint> key_point_vec = GetKeyPoints(response_layer);

    cout << "key_point_vec:" << key_point_vec.size() << endl;

    return key_point_vec;
}
