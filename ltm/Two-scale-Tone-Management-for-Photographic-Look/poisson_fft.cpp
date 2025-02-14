#include "poisson_fft.hpp"

MyPoissonFusionTest::MyPoissonFusionTest() {
}

MyPoissonFusionTest::~MyPoissonFusionTest() {
}

void MyPoissonFusionTest::computeLaplacianX(Mat img, Mat &laplacianX) {
    Mat kernel = Mat::zeros(1, 3, CV_32F);
    kernel.at<float>(0,0) = -1;
    kernel.at<float>(0,1) = 1;
    filter2D(img, laplacianX, CV_32F, kernel);
}

void MyPoissonFusionTest::computeLaplacianY(Mat img, Mat &laplacianY) {
    Mat kernel = Mat::zeros(3, 1, CV_32F);
    kernel.at<float>(0,0) = -1;
    kernel.at<float>(1,0) = 1;
    filter2D(img, laplacianY, CV_32F, kernel);
}

void MyPoissonFusionTest::dst(const cv::Mat& src, Mat& dest, bool invert) {
    Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

    int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE: DFT_ROWS;
    src.copyTo(temp(Rect(1,0, src.cols, src.rows)));

    for(int j = 0 ; j < src.rows ; ++j) {
        float * tempLinePtr = temp.ptr<float>(j);
        const float * srcLinePtr = src.ptr<float>(j);
        for(int i = 0 ; i < src.cols ; ++i) {
            tempLinePtr[src.cols + 2 + i] = - srcLinePtr[src.cols - 1 - i];
        }
    }

    Mat planes[] = {temp, Mat::zeros(temp.size(), CV_32F)};
    Mat complex;

    merge(planes, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes);
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

    for(int j = 0 ; j < src.cols ; ++j) {
        float * tempLinePtr = temp.ptr<float>(j);
        for(int i = 0 ; i < src.rows ; ++i) {
            float val = planes[1].ptr<float>(i)[j + 1];
            tempLinePtr[i + 1] = val;
            tempLinePtr[temp.cols - 1 - i] = - val;
        }
    }

    Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_32F)};

    merge(planes2, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes2);

    temp = planes2[1].t();
    temp(Rect(0, 1, src.cols, src.rows)).copyTo(dest);
}

void MyPoissonFusionTest::solve(Mat &img, Mat& mod_diff, Mat &result) {
    const int w = img.cols;
    const int h = img.rows;

	vector<float> filter_X, filter_Y;
    initVariables(img, filter_X, filter_Y);

    Mat res;
    dst(mod_diff, res,false);

    for(int j = 0 ; j < h-2; j++) {
        float * resLinePtr = res.ptr<float>(j);
        for(int i = 0 ; i < w-2; i++) {
            resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
        }
    }

    dst(res, mod_diff, true);

    float * resLinePtr = result.ptr<float>(0);
    float * imgLinePtr = img.ptr<float>(0);
    const float * interpLinePtr = NULL;

    for(int i = 0 ; i < w ; ++i)
        result.ptr<float>(0)[i] = img.ptr<float>(0)[i];

    for(int j = 1 ; j < h-1 ; ++j) {
        resLinePtr = result.ptr<float>(j);
        imgLinePtr  = img.ptr<float>(j);
        interpLinePtr = mod_diff.ptr<float>(j-1);

        resLinePtr[0] = imgLinePtr[0];

        for(int i = 1 ; i < w-1 ; ++i) {
            float value = interpLinePtr[i-1];
            resLinePtr[i] = static_cast<float>(value);
        }
        resLinePtr[w-1] = imgLinePtr[w-1];
    }

    resLinePtr = result.ptr<float>(h-1);
    imgLinePtr = img.ptr<float>(h-1);
    for(int i = 0 ; i < w ; ++i)
        resLinePtr[i] = imgLinePtr[i];
}

Mat MyPoissonFusionTest::poissonSolver(Mat img, Mat laplacianX , Mat laplacianY) {
    int w = img.cols;
    int h = img.rows;

    Mat lap = laplacianX + laplacianY;
    Mat bound = img.clone();

    rectangle(bound, Point(1, 1), Point(img.cols-2, img.rows-2), Scalar::all(0), -1);
    Mat boundary_points;
    Laplacian(bound, boundary_points, CV_32F);

    boundary_points = lap - boundary_points;

    Mat mod_diff = boundary_points(Rect(1, 1, w-2, h-2));

    img.convertTo(img, CV_32FC1);
	Mat result = img.clone();
    solve(img,mod_diff,result);

	return result;
}

void MyPoissonFusionTest::initVariables(Mat destination, vector<float> &filter_X, vector<float> &filter_Y) {
    const int w = destination.cols;
    filter_X.resize(w - 2);
    double scale = CV_PI / (w - 1);
    for(int i = 0 ; i < w-2 ; ++i)
        filter_X[i] = 2.0f * (float)std::cos(scale * (i + 1));

	const int h  = destination.rows;
	filter_Y.resize(h - 2);
	scale = CV_PI / (h - 1);
	for(int j = 0 ; j < h - 2 ; ++j)
		filter_Y[j] = 2.0f * (float)std::cos(scale * (j + 1));
}

Mat MyPoissonFusionTest::poisson(Mat destination, Mat patchGradientX, Mat patchGradientY) {
	computeLaplacianX(patchGradientX, patchGradientX);
	computeLaplacianY(patchGradientY, patchGradientY);

	Mat out = poissonSolver(destination, patchGradientX, patchGradientY);

	return out;
}

Mat MyPoissonFusionTest::Run(Mat src, Mat patchGradientX, Mat patchGradientY) {

    Mat out = poisson(src, patchGradientX, patchGradientY);

    return out;
}
