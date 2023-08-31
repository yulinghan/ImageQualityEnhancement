#include "MTBFramesRegistration.hpp"

MyMtbTest::MyMtbTest() {
}

MyMtbTest::~MyMtbTest() {
}

void MyMtbTest::BuildPyr(const Mat& img, std::vector<Mat>& pyr) {
    pyr.push_back(img.clone());
    for (int level = 0; level < max_level_; level++) {
        Mat cur_mat;
        resize(pyr[level], cur_mat, pyr[level].size()/2);
        pyr.push_back(cur_mat);
    }
}

int MyMtbTest::GetAverage(Mat& img) {
    unsigned long long sum = 0, area;
    int avg;
    int rows = img.rows;
    int cols = img.cols;
    area = rows*cols;
    for (int i = 0; i < rows; i++) {
        const uchar* inData = img.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            sum = sum + inData[j];
        }
    }
    avg = (int)(sum / area);
    return avg;
}

int MyMtbTest::GetMedian(Mat& img) {
    int channels = 0;
    Mat hist;
    int hist_size = LDR_SIZE;
    float range[] = { 0, LDR_SIZE };
    const float* ranges[] = { range };
    calcHist(&img, 1, &channels, Mat(), hist, 1, &hist_size, ranges);
    float *ptr = hist.ptr<float>();
    int median = 0, sum = 0;
    int thresh = (int)img.total() / 2;
    while (sum < thresh && median < LDR_SIZE) {
        sum += static_cast<int>(ptr[median]);
        median++;
    }
    return median;
}

void MyMtbTest::ComputeBitmaps(InputArray _img, OutputArray _tb) {
    Mat img = _img.getMat();
    _tb.create(img.size(), CV_8U);
    Mat tb = _tb.getMat();
    int median = GetMedian(img);
    compare(img, median, tb, CMP_GT);
}

Mat MyMtbTest::ShiftMat(Mat src, Point shift) {
    Mat dst = Mat::zeros(src.size(),src.type());
    int width = src.cols - abs(shift.x);
    int height = src.rows - abs(shift.y);
    Rect dst_rect(max(-shift.x, 0), max(-shift.y, 0), width, height);
    Rect src_rect(max(shift.x, 0), max(shift.y, 0), width, height);
    src(src_rect).copyTo(dst(dst_rect));

    return dst;
}

Point MyMtbTest::CalculateShift(const Mat& img, const vector<Mat> tbRef, const Rect& ROI) {
    Point shift(0, 0);

    Mat imgROI = img(ROI);
    GaussianBlur(imgROI, imgROI, Size(3,3), 0);
    vector<Mat> pyr;
    BuildPyr(imgROI, pyr);

    for (int level = max_level_; level >= 1; level--) {
        int levelRef = max_level_ - level;
        shift = shift * 2;
        Mat tbImg;
        ComputeBitmaps(pyr[level], tbImg);

        Point point0(0, 0), point1(-1, 0), point2(0, 1);
        Point point3(1, 0), point4(0, -1), point5(-1, 1);
        Point point6(1, 1), point7(1, -1), point8(-1, -1);
        Point points[9] = { point0, point1, point2, point3, point4, point5, point6, point7, point8 };
        Point test_shift[9] = { point0, point0, point0, point0, point0, point0, point0, point0, point0 };
        int min_ID = 0;

        int err[9];
        for (int i = 0; i < 9; i++) {
            test_shift[i] = shift + points[i];
            Mat diff;
            int width = tbRef[levelRef].cols - abs(test_shift[i].x);
            int height = tbRef[levelRef].rows - abs(test_shift[i].y);
            Rect dst_rect(max(test_shift[i].x, 0), max(test_shift[i].y, 0), width, height);
            Rect src_rect(max(-test_shift[i].x, 0), max(-test_shift[i].y, 0), width, height);
            bitwise_xor(tbRef[levelRef](src_rect), tbImg(dst_rect), diff);
            err[i] = countNonZero(diff);
        }
        int min_err = err[0];
        for (int i = 1; i < 9; i++) {
            if (err[i] < min_err) {
                min_err = err[i];
            }
        }
        for (int i = 1; i < 9; i++) {
            if (err[i] == min_err) {
                min_ID = i;
                break;
            }
        }

        shift = test_shift[min_ID];
    }
    return shift*2;
}

Mat MyMtbTest::RegistrationY(Mat image, vector<Mat> tbRef, Rect ROI) {
    Point Shift = CalculateShift(image, tbRef, ROI);
    printf("shift: %d,%d \n", Shift.x, Shift.y);
    Mat warped = ShiftMat(image, Shift);

    return warped;
}

Mat MyMtbTest::Run(Mat input, Mat ref) {
    int rows = ref.rows, cols = ref.cols;
    Rect ROI((cols - roi_length_) / 2, (rows - roi_length_) / 2, roi_length_, roi_length_);
    Mat refROI = ref(ROI);
    GaussianBlur(refROI, refROI, Size(3,3), 0);
    vector<Mat> tbRef, pyrRef;
    BuildPyr(refROI, pyrRef);
    for (int level = max_level_; level >= 1; level--) {
        Mat tb;
        ComputeBitmaps(pyrRef[level], tb);
        tbRef.push_back(tb);
    }

    Mat warped = RegistrationY(input, tbRef, ROI);

    return warped;
}
