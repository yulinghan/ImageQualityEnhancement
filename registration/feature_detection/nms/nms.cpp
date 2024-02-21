#include "nms.hpp"

MyNmsTest::MyNmsTest() {
}

MyNmsTest::~MyNmsTest() {
}

void MyNmsTest::GetGradient(Mat src, Mat &Ixx, Mat &Ixy, Mat &Iyy){
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

Mat MyNmsTest::CalcMinEigenVal(Mat Ixx, Mat Ixy, Mat Iyy) {
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

Mat MyNmsTest::CornersShow(Mat src, vector<KeyPoint> corners) {
    Mat dst = src.clone();

    for(int i=0; i<corners.size(); i++) {
        circle(dst, corners[i].pt, 3, Scalar(255, 0, 0), -1);  // 画半径为1的圆(画点）
    }

    return dst;
}

vector<float*> MyNmsTest::CornersChoice(Mat eig, float thresh) {
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

void MyNmsTest::DistanceChoice(float minDistance, int maxCorners, vector<KeyPoint> &corners) {
    int ncorners = 0;
    Mat eig = eig_;
    vector<float*> tmpCorners = tmpCorners_;

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
                KeyPoint kp;
                kp.pt = Point(x, y);
                kp.response = *tmpCorners[i];
                corners.push_back(kp);
                ncorners += 1;
                if(maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
}

void MyNmsTest::GetKeyPoint(Mat eig, vector<float*> tmpCorners, int maxCorners, vector<KeyPoint> &corners) {
    int ncorners = 0;
    for(int i = 0; i < eig.rows*eig.cols; i++) {
        int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
        int y = (int)(ofs / eig.step);
        int x = (int)((ofs - y * eig.step) / sizeof(float));

        KeyPoint kp;
        kp.pt = Point2f((float)x, (float)y);
        kp.response = *tmpCorners[i];
        corners.push_back(kp);

        ncorners += 1;
        if( maxCorners > 0 && (int)ncorners == maxCorners )
            break;
    }
}

vector<KeyPoint> MyNmsTest::run(Mat src, int maxCorners) {
    src.convertTo(src, CV_32FC1, 1/255.0);

    //计算图像梯度提取
    Mat Ixx, Ixy, Iyy;
    GetGradient(src, Ixx, Ixy, Iyy);

    int kern_size = 5;
    GaussianBlur(Ixx, Ixx, Size(kern_size, kern_size), 0, 0);
    GaussianBlur(Ixy, Ixy, Size(kern_size, kern_size), 0, 0);
    GaussianBlur(Iyy, Iyy, Size(kern_size, kern_size), 0, 0);

    //计算Shi-Tomasi交点评分:矩阵M的较小特征值
    eig_ = CalcMinEigenVal(Ixx, Ixy, Iyy);

    //根据阈值对角点筛选和排序
    float thresh = 0.01;
    tmpCorners_ = CornersChoice(eig_, thresh);

    vector<KeyPoint> corners;
    GetKeyPoint(eig_, tmpCorners_, maxCorners, corners);

    return corners;
}

double MyNmsTest::computeR(Point2i x1, Point2i x2) {
    return norm(x1 - x2);
}

template <typename T> vector<size_t>  MyNmsTest::sort_indexes(const vector<T> &v) {
    // initialize original index locations
    vector< size_t>  idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) {return v[i1] >  v[i2]; });
    return idx;
}

vector<KeyPoint> MyNmsTest::ANMS(vector<KeyPoint> kpts, int num) {
    int sz = kpts.size();
    double maxmum = 0;
    vector<double> roblocalmax(kpts.size());
    vector<double> raduis(kpts.size(), INFINITY);
    for (size_t i = 0; i < sz; i++) {
        auto rp = kpts[i].response;
        if (rp > maxmum)
            maxmum = rp;
        roblocalmax[i] = rp*0.9;
    }
    auto max_response = maxmum*0.9;
    for (size_t i = 0; i < sz; i++) {
        double rep = kpts[i].response;
        Point2i p = kpts[i].pt;
        auto& rd = raduis[i];
        if (rep>max_response) {
            rd = INFINITY;
        } else {
            for (size_t j = 0; j < sz; j++) {
                if (roblocalmax[j] > rep) {
                    auto d = computeR(kpts[j].pt, p);
                    if (rd > d)
                        rd = d;
                }
            }
        }
    }
    auto sorted = sort_indexes(raduis);
    vector<KeyPoint> rpts;

    for (size_t i = 0; i < num; i++) {
        rpts.push_back(kpts[sorted[i]]);
    }
    return std::move(rpts);
}

struct KeypointResponseGreater {
    inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const {
        return kp1.response > kp2.response;
    }
};

vector<KeyPoint> MyNmsTest::ssc(vector<KeyPoint> keyPoints, int numRetPoints, float tolerance, int cols, int rows) {
    sort(keyPoints.begin(), keyPoints.end(), KeypointResponseGreater());

    // several temp expression variables to simplify solution equation
    int exp1 = rows + cols + 2 * numRetPoints;
    long long exp2 =
        ((long long)4 * cols + (long long)4 * numRetPoints +
         (long long)4 * rows * numRetPoints + (long long)rows * rows +
         (long long)cols * cols - (long long)2 * rows * cols +
         (long long)4 * rows * cols * numRetPoints);
    double exp3 = sqrt(exp2);
    double exp4 = numRetPoints - 1;

    double sol1 = -round((exp1 + exp3) / exp4); // first solution
    double sol2 = -round((exp1 - exp3) / exp4); // second solution

    // binary search range initialization with positive solution
    int high = (sol1 > sol2) ? sol1 : sol2;
    int low = floor(sqrt((double)keyPoints.size() / numRetPoints));
    low = max(1, low);

    int width;
    int prevWidth = -1;

    vector<int> ResultVec;
    bool complete = false;
    unsigned int K = numRetPoints;
    unsigned int Kmin = round(K - (K * tolerance));
    unsigned int Kmax = round(K + (K * tolerance));

    vector<int> result;
    result.reserve(keyPoints.size());
    while (!complete) {
        width = low + (high - low) / 2;
        if (width == prevWidth || low > high) { // needed to reassure the same radius is not repeated again
            ResultVec = result; // return the keypoints from the previous iteration
            break;
        }
        result.clear();

        double c = (double)width / 2.0; // initializing Grid
        int numCellCols = floor(cols / c);
        int numCellRows = floor(rows / c);
        vector<vector<bool>> coveredVec(numCellRows + 1, vector<bool>(numCellCols + 1, false));

        for(unsigned int i = 0; i < keyPoints.size(); ++i) {
            int row = floor(keyPoints[i].pt.y / c); // get position of the cell current point is located at
            int col = floor(keyPoints[i].pt.x / c);
            if (coveredVec[row][col] == false) { // if the cell is not covered
                result.push_back(i);
                int rowMin = ((row - floor(width / c)) >= 0) ? (row - floor(width / c)) : 0; // get range which current radius is covering
                int rowMax = ((row + floor(width / c)) <= numCellRows) ? (row + floor(width / c)) : numCellRows;
                int colMin = ((col - floor(width / c)) >= 0) ? (col - floor(width / c)) : 0;
                int colMax = ((col + floor(width / c)) <= numCellCols) ? (col + floor(width / c)) : numCellCols;
                for (int rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov) {
                    for (int colToCov = colMin; colToCov <= colMax; ++colToCov) {
                        if (!coveredVec[rowToCov][colToCov]) {
                            coveredVec[rowToCov][colToCov] = true; // cover cells within the square bounding box with width
                        }
                    }
                }
            }
        }

        if (result.size() >= Kmin && result.size() <= Kmax) { // solution found
            ResultVec = result;
            complete = true;
        } else if (result.size() < Kmin)
            high = width - 1; // update binary search range
        else
            low = width + 1;
        prevWidth = width;
    }
    // retrieve final keypoints
    vector<cv::KeyPoint> kp;
    for (unsigned int i = 0; i < ResultVec.size(); i++)
        kp.push_back(keyPoints[ResultVec[i]]);

    return kp;
}
