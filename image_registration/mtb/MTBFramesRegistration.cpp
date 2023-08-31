#include "MTBFramesRegistration.hpp"

MyMtbTest::MyMtbTest() {
}

MyMtbTest::~MyMtbTest() {
}

Mat MyMtbTest::SparseBlur(Mat src, int sparseFactor) {
    int winSize = 2 * sparseFactor + 3;
    int boundary = (winSize - 1) / 2;
    Mat dst = Mat::zeros(src.size(), src.type());
    for (int i = boundary; i < src.rows - boundary; i++) {
        const uchar* lastData = src.ptr<uchar>(i - 1 - sparseFactor);
        const uchar* curData = src.ptr<uchar>(i);
        const uchar* nextData = src.ptr<uchar>(i + 1 + sparseFactor);
        uchar* outData = dst.ptr<uchar>(i);
        for (int j = boundary; j < src.cols - boundary; j++) {
            outData[j] = (lastData[j - 1 - sparseFactor] + 2 * lastData[j] + lastData[j + 1 + sparseFactor] +
                    2 * curData[j - 1 - sparseFactor] + 4 * curData[j] + 2 * curData[j + 1 + sparseFactor] +
                    nextData[j - 1 - sparseFactor] + 2 * nextData[j] + nextData[j + 1 + sparseFactor]) / 16;
        }
    }
    return dst;
}

void MyMtbTest::Downsample(Mat& src, Mat& dst) {
    dst = Mat(src.rows / 2, src.cols / 2, CV_8UC1);

    int offset = src.cols * 2;
    uchar *src_ptr = src.ptr();
    uchar *dst_ptr = dst.ptr();
    for (int y = 0; y < dst.rows; y++) {
        uchar *ptr = src_ptr;
        for (int x = 0; x < dst.cols; x++) {
            dst_ptr[0] = ptr[0];
            dst_ptr++;
            ptr += 2;
        }
        src_ptr += offset;
    }
}

void MyMtbTest::BuildPyr(const Mat& img, std::vector<Mat>& pyr) {
    pyr.resize(max_level_ + 1);
    pyr[0] = img.clone();
    for (int level = 0; level < max_level_; level++) {
        Downsample(pyr[level], pyr[level + 1]);
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
    Mat imgROIblur = SparseBlur(imgROI, 3);
    GaussianBlur(imgROIblur, imgROIblur, Size(3,3), 0);
    vector<Mat> pyr;
    BuildPyr(imgROIblur, pyr);

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

        if (level == 1) {
            int err[5];
            for (int i = 0; i < 5; i++) {
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
            for (int i = 1; i < 5; i++) {
                if (err[i] < min_err) {
                    min_err = err[i];
                }
            }
            for (int i = 1; i < 5; i++) {
                if (err[i] == min_err) {
                    min_ID = i;
                    break;
                }
            }
        } else {
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
    Mat refROIblur = SparseBlur(refROI, 3);
    GaussianBlur(refROIblur, refROIblur, Size(3,3), 0);
    vector<Mat> tbRef, pyrRef;
    BuildPyr(refROIblur, pyrRef);
    for (int level = max_level_; level >= 1; level--) {
        Mat tb;
        ComputeBitmaps(pyrRef[level], tb);
        tbRef.push_back(tb);
    }

    Mat warped = RegistrationY(input, tbRef, ROI);

    return warped;
}
