#include "shi_tomasi.hpp"

MyShiTomasiTest::MyShiTomasiTest() {
}

MyShiTomasiTest::~MyShiTomasiTest() {
}

void MyShiTomasiTest::GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy){
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

Mat MyShiTomasiTest::CalcMinEigenVal(Mat Ixx, Mat Ixy, Mat Iyy) {
    Mat res = Mat::zeros(Ixx.size(),CV_32FC1);

    for(int i = 0; i < Ixx.rows; i++) {
        for(int j = 0; j < Ixx.cols; j++) {
            float a = Ixx.at<float>(i, j);
            float b = Ixy.at<float>(i, j);
            float c = Iyy.at<float>(i, j);

            res.at<float>(i,j) = (a+c) - sqrt((a-c)*(a-c) + b*b);
        }
    }
    return res;
}

vector<Point> MyShiTomasiTest::get_corners(Mat res, float thresh) {
    vector<Point> corners;
    for (int r=0; r<res.rows; r++) {
        for(int c=0; c<res.cols; c++) { 
            if(res.at<float>(r,c)>thresh) {
                corners.emplace_back(c,r);
            }
        }
    }

    return corners;
}

Mat MyShiTomasiTest::CornersShow(Mat src, vector<Point> corners) {
    Mat dst = src.clone();

    for(int i=0; i<corners.size(); i++) {
        circle(dst, corners[i], 3, Scalar(255, 0, 0), -1);  // 画半径为1的圆(画点）
    }

    return dst;
}

vector<float*> MyShiTomasiTest::CornersChoice(Mat eig, float thresh) {
    vector<float*> tmpCorners;

    double max_val = 0;
    minMaxLoc(eig, 0, &max_val, 0, 0);
    threshold(eig, eig, max_val*thresh, 0, THRESH_TOZERO);

    Mat tmp;
    dilate(eig, tmp, cv::Mat());

    for(int y = 0; y < eig.rows; y++){
        float* eig_data = (float*)eig.ptr(y);
        float* tmp_data = (float*)tmp.ptr(y);

        for(int x = 0; x < eig.cols; x++){
            float val = eig_data[x];
            if(val != 0 && val == tmp_data[x]) {
                tmpCorners.push_back(eig_data + x);
            }
        }
    }

    sort(tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

    return tmpCorners;
}

void MyShiTomasiTest::DistanceChoice(Mat eig, vector<float*> tmpCorners, float minDistance, int maxCorners,
            vector<Point> &corners, vector<float> &scores) {

    int ncorners = 0;

    if (minDistance >= 1) {
        int w = eig.cols;
        int h = eig.rows;

        int cell_size = cvRound(minDistance);
        int grid_width = (w + cell_size - 1) / cell_size;
        int grid_height = (h + cell_size - 1) / cell_size;

        vector<vector<Point2f>> grid(grid_width * grid_height);
        double minDistance_squre = minDistance * minDistance;

        for(int i = 0; i < eig.rows*eig.cols; i++) {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for(int yy = y1; yy <= y2; yy++) {
                for(int xx = x1; xx <= x2; xx++){
                    vector<Point2f> &m = grid[yy * grid_width + xx];
                    if(m.size()) {
                        for(int j = 0; j < m.size(); j++) {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if(dx*dx + dy*dy < minDistance_squre) {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }
break_out:

            if(good) {
                grid[y_cell*grid_width + x_cell].push_back(Point(x, y));

                corners.push_back(Point(x, y));
                scores.push_back(*tmpCorners[i]);
                ncorners += 1;

                if(maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    } else {
        for(int i = 0; i < eig.rows*eig.cols; i++) {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y * eig.step) / sizeof(float));

            corners.push_back(cv::Point2f((float)x, (float)y));
            scores.push_back(*tmpCorners[i]);

            ncorners += 1;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }
}

vector<cv::Point> MyShiTomasiTest::run(Mat src) {
    src.convertTo(src, CV_32FC1, 1/255.0);

    //计算图像梯度提取
    Mat Ixx, Ixy, Iyy;
    GetGradient(src, Ixx, Ixy, Iyy);

    int kern_size = 5;
    GaussianBlur(Ixx, Ixx, Size(kern_size, kern_size), 0, 0);
    GaussianBlur(Ixy, Ixy, Size(kern_size, kern_size), 0, 0);
    GaussianBlur(Iyy, Iyy, Size(kern_size, kern_size), 0, 0);

    //计算Shi-Tomasi交点评分:矩阵M的较小特征值
    Mat eig = CalcMinEigenVal(Ixx, Ixy, Iyy);

    //根据阈值对角点筛选和排序
    float thresh = 0.01;
    vector<float*> tmpCorners = CornersChoice(eig, thresh);

    //设置角点之间最小间隔。
    //选取预设数量的最佳特征点
    float minDistance = 3.0;
    int maxCorners = 500;
    vector<Point> corners;
    vector<float> scores;
    DistanceChoice(eig, tmpCorners, minDistance, maxCorners, corners, scores);

    return corners;
}
