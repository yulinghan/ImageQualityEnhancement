#include "PoissonMatting.hpp"
#include <numeric>

PoissonMatting::PoissonMatting() {
}

PoissonMatting::~PoissonMatting() {
}

double dist_sqr(Point p1, Point p2) {
    return (p1.x - p2.x) *(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}
//颜色距离，就是两个颜色的最大值相减
int color_dis(cv::Vec3b p1, cv::Vec3b p2) {
    int t1 = fmax(fmax(p1[0], p1[1]), p1[2]);
    int t2 = fmax(fmax(p2[0], p2[1]), p2[2]);
    return t1 - t2;
}
//防止x,y越界
int inX(cv::Mat &image, int x) {
    if (x < 0) x = 0;
    if (x >= image.cols) x = image.cols - 1;
    return x;
}
int inY(cv::Mat &image, int y) {
    if (y < 0) y = 0;
    if (y >= image.rows) y = image.rows - 1;
    return y;
}
//intensity，三通道的最大值
double intensity(Vec3b v) {  
    return fmax(fmax(v[0], v[1]), v[2]);
}
/***************************************************/

//找出所有边界像素的位置
vector<Point> PoissonMatting::findBoundaryPixels(Mat trimap, int a, int b){
    vector<Point> result;

    for (int x = 1; x < trimap.cols - 1; ++x) {
        for (int y = 1; y < trimap.rows - 1; ++y) {
            if (trimap.at<uchar>(y, x) == a) {
                if (trimap.at<uchar>(y-1, x)==b || trimap.at<uchar>(y+1, x)==b || trimap.at<uchar>(y, x-1)==b || trimap.at<uchar>(y, x+1)==b) {
                    result.push_back(Point(x, y));
                }
            }
        }
    }
    return result;
}

Mat PoissonMatting::matting(Mat image, Mat trimap){
#define I(x, y) (image.at<uchar>(inY(image, y), inX(image, x)))
#define FmB(y, x) (FminusB.at<double>(inY(FminusB, y), inX(FminusB, x)))
    Mat alpha;
    alpha.create(image.size(), CV_8UC1);

    Mat FminusB = Mat::zeros(trimap.size(), CV_64FC1);

    for (int times = 0; times < 5; ++times) {
        //传入的trimap图上,前景值是255, 背景值是0, 未知区域为128.
        vector<Point> foregroundBoundary = findBoundaryPixels(trimap, 255, 128);
        vector<Point> backgroundBoundary = findBoundaryPixels(trimap, 0, 128);

        Mat trimap_blur;
        GaussianBlur(trimap, trimap_blur, Size(9, 9), 0);

        // 构建图像上每个位置的 F-B
        for (int x = 0; x < trimap.cols; ++x) {
            for (int y = 0; y < trimap.rows; ++y) {
                Point current;
                current.x = x;
                current.y = y;
                if (trimap_blur.at<uchar>(y, x) == 255) {//确定的前景部分F-(0,0,0)
                    FminusB.at<double>(y, x) = image.at<uchar>(y, x); 
                } else if (trimap_blur.at<uchar>(y, x) == 0) {//确定的背景部分(0,0,0)-B
                    FminusB.at<double>(y, x) = -image.at<uchar>(y, x);
                } else {
                    // 未知区域的每个位置寻找距离最近的前景像素位置和背景像素位置
                    Point nearestForegroundPoint, nearestBackgroundPoint;
                    double nearestForegroundDistance = 1e9, nearestBackgroundDistance = 1e9;
                    for(Point &p : foregroundBoundary) {
                        double t = dist_sqr(p, current);
                        if (t < nearestForegroundDistance) {
                            nearestForegroundDistance = t;
                            nearestForegroundPoint = p;
                        }
                    }
                    for(Point &p : backgroundBoundary) {
                        double t = dist_sqr(p, current);
                        if (t < nearestBackgroundDistance) {
                            nearestBackgroundDistance = t;
                            nearestBackgroundPoint = p;
                        }
                    }
                    
                    FminusB.at<double>(y, x) = image.at<uchar>(nearestForegroundPoint.y, nearestForegroundPoint.x)
                                                - image.at<uchar>(nearestBackgroundPoint.y, nearestBackgroundPoint.x);
                                                               
                    if (FminusB.at<double>(y, x) == 0)
                        FminusB.at<double>(y, x) = 1e-9;
                }
            }
        }

        // F-B高斯平滑
        GaussianBlur(FminusB, FminusB, Size(9, 9), 0);
        // Solve the Poisson Equation By The Gauss-Seidel Method (Iterative Method)
        for (int times2 = 0; times2 < 300; ++times2) {
            for (int x = 0; x < trimap.cols; ++x) {
                for (int y = 0; y < trimap.rows; ++y) {

                    //白色(F) 黑色(B) 灰色(未知)
                    if (trimap.at<uchar>(y, x) == 128) {  
                        // 计算 (▽I/F-B)在所有位置的散度dvgX:(u/v)' = (u'v-uv')/v^2
                        double dvgX = ((I(x+1,y) + I(x-1,y) - 2*I(x,y)) * FmB(y,x)
                                - (I(x+1,y) - I(x,y))*(FmB(y,x+1) - FmB(y,x)))
                                / (FmB(y, x) * FmB(y, x));
                        // 计算 (▽I/F-B)在所有位置的散度dvgY:(u/v)' = (u'v-uv')/v^2
                        double dvgY = ((I(x,y+1) + I(x,y-1) - 2*I(x, y)) * FmB(y, x)
                                - (I(x, y+1) - I(x,y)) * (FmB(y+1, x) - FmB(y, x)))
                                / (FmB(y, x) * FmB(y, x));
                        double dvg = dvgX + dvgY;

                        // a的散度为：a(i+1,j)+a(i-1,j)+a(i,j+1)+a(i,j-1)-4a(i,j) = dvg, 因此得到下面的式子
                        double newAlpha = ((double)alpha.at<uchar>(y, x + 1)
                                        + alpha.at<uchar>(y, x - 1)
                                        + alpha.at<uchar>(y + 1, x)
                                        + alpha.at<uchar>(y - 1, x)
                                        - dvg * 255.0) / 4.0;
                        // 根据得到的alpha更新Trimap
                        if (newAlpha > 253)
                            trimap.at<uchar>(y, x) = 255;
                        else if (newAlpha < 3)  
                            trimap.at<uchar>(y, x) = 0;
                        if (newAlpha < 0)
                            newAlpha = 0;
                        if (newAlpha > 255)
                            newAlpha = 255;
                        // 更新alpha
                        alpha.at<uchar>(y, x) = newAlpha;
                    } else if (trimap.at<uchar>(y, x) == 255) {
                        alpha.at<uchar>(y, x) = 255;   //前景区域不需计算
                    } else if (trimap.at<uchar>(y, x) == 0) {
                        alpha.at<uchar>(y, x) = 0;     //背景区域不需计算
                    }
                }
            }
        }
    }

    return alpha;
}

Mat PoissonMatting::Run(Mat src, Mat alpha_mat) {
    Mat out = matting(src, alpha_mat);

    return out;
}
