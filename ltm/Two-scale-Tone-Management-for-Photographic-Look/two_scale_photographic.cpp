#include "two_scale_photographic.hpp"
#include "poisson_fft.hpp"
#include <numeric>

TwoScalePhotoGraphic::TwoScalePhotoGraphic() {
}

TwoScalePhotoGraphic::~TwoScalePhotoGraphic() {
}

// 计算图像的直方图
cv::Mat calculateHistogram(const cv::Mat& image, int max_value) {
    cv::Mat histogram;
    int histSize = max_value+1; // 直方图的bin数量
    float range[] = {0, histSize}; // 像素值范围
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;

    // 计算直方图
    cv::calcHist(&image, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange, uniform, accumulate);
    return histogram;
}

// 计算累积直方图
cv::Mat calculateCumulativeHistogram(const cv::Mat& histogram) {
    cv::Mat cumulativeHistogram = histogram.clone();
    for (int i = 1; i < histogram.rows; ++i) {
        cumulativeHistogram.at<float>(i, 0) += cumulativeHistogram.at<float>(i - 1, 0);
    }
    return cumulativeHistogram;
}

// 执行直方图匹配
cv::Mat histogramMatching(const cv::Mat& sourceImage, const cv::Mat& targetImage, int max_value) {

    // 计算源图像和目标图像的直方图
    cv::Mat sourceHistogram = calculateHistogram(sourceImage, max_value);
    cv::Mat targetHistogram = calculateHistogram(targetImage, max_value);

    // 计算源图像和目标图像的累积直方图
    cv::Mat sourceCumulativeHistogram = calculateCumulativeHistogram(sourceHistogram);
    cv::Mat targetCumulativeHistogram = calculateCumulativeHistogram(targetHistogram);

    // 归一化累积直方图
    sourceCumulativeHistogram /= sourceCumulativeHistogram.at<float>(max_value, 0);
    targetCumulativeHistogram /= targetCumulativeHistogram.at<float>(max_value, 0);

    // 创建查找表
    cv::Mat lookupTable(1, max_value, CV_16U);
    for (int i = 0; i < max_value; i++) {
        // 使用二分查找找到最接近的匹配值
        auto it = lower_bound(targetCumulativeHistogram.ptr<float>(0), targetCumulativeHistogram.ptr<float>(max_value) + 2, sourceCumulativeHistogram.at<float>(i, 0));
        int index = std::distance(targetCumulativeHistogram.ptr<float>(0), it);
        lookupTable.at<ushort>(0, i) = static_cast<ushort>(index);
    }

    // 应用查找表到源图像
    cv::Mat matchedImage = sourceImage.clone();
    for(int i=0; i<sourceImage.rows; i++) {
        for(int j=0; j<sourceImage.cols; j++) {
            matchedImage.at<ushort>(i, j) = lookupTable.at<ushort>(matchedImage.at<ushort>(i, j));
        }
    }

    return matchedImage;
}

void GetTowScaleImg(Mat src, Mat ref, Mat &base_src, Mat &detail_src, Mat &base_ref, Mat &detail_ref) {

     //双边滤波参数
    int filterDiameter = 5;  // 滤波器直径
    double colorSigma = 30;  // 颜色空间标准差
    double spaceSigma = 70;  // 空间域标准差

    //双边滤波
    bilateralFilter(src, base_src, filterDiameter, colorSigma, spaceSigma);
    detail_src = src - base_src;

    bilateralFilter(ref, base_ref, filterDiameter, colorSigma, spaceSigma);
    detail_ref = ref - base_ref;
}

Mat GradientReversalRemoval(Mat src, Mat detail_src) {
    Mat gxI = Mat::zeros(src.size(), CV_32FC1);
    Mat gyI = Mat::zeros(src.size(), CV_32FC1);
    Mat gxD = Mat::zeros(src.size(), CV_32FC1);
    Mat gyD = Mat::zeros(src.size(), CV_32FC1);
    Mat gx  = Mat::zeros(src.size(), CV_32FC1);
    Mat gy  = Mat::zeros(src.size(), CV_32FC1);

    for(int i=0; i<src.rows-1; i++) {
        for(int j=0; j<src.cols-1; j++) {
            gxI.at<float>(i, j) = src.at<float>(i, j+1) - src.at<float>(i, j);
            gyI.at<float>(i, j) = src.at<float>(i+1, j) - src.at<float>(i, j);
            gxD.at<float>(i, j) = detail_src.at<float>(i, j+1) - detail_src.at<float>(i, j);
            gyD.at<float>(i, j) = detail_src.at<float>(i+1, j) - detail_src.at<float>(i, j);
        }
    }

    for(int i=0; i<src.rows; i++) {
        float *ptr_gxD = gxD.ptr<float>(i);
        float *ptr_gyD = gyD.ptr<float>(i);
        float *ptr_gxI = gxI.ptr<float>(i);
        float *ptr_gyI = gyI.ptr<float>(i);
        float *ptr_gx  = gx.ptr<float>(i);
        float *ptr_gy  = gy.ptr<float>(i);
        for(int j=0; j<src.cols; j++) {
            if((ptr_gxD[j]>0 && ptr_gxI[j]<0) || (ptr_gxD[j]<0 && ptr_gxI[j]>0)) {
                ptr_gx[j] = 0;
            } else if(abs(ptr_gxD[j]) > abs(ptr_gxI[j])) {
                ptr_gx[j] = ptr_gxI[j];
            } else {
                ptr_gx[j] = ptr_gxD[j];
            }

            if((ptr_gyD[j]>0 && ptr_gyI[j]<0) || (ptr_gyD[j]<0 && ptr_gyI[j]>0)) {
                ptr_gy[j] = 0;
            } else if(abs(ptr_gyD[j]) > abs(ptr_gyI[j])) {
                ptr_gy[j] = ptr_gyI[j];
            } else {
                ptr_gy[j] = ptr_gyD[j];
            }
        }
    }
 
    MyPoissonFusionTest *my_poisson_test = new MyPoissonFusionTest();
    Mat out = my_poisson_test->Run(detail_src, gx, gy);

    return out;
}

Mat CalGaussianTemplate(int r, float sigma) {
    float pi = 3.1415926;
    int center = r;
    int ksize = r*2+1;
    float x2, y2;

    Mat Kore = Mat::zeros(Size(ksize, ksize), CV_32FC1);
    for (int i = 0; i < ksize; i++) {
        x2 = pow(i - center, 2);
        for (int j = 0; j < ksize; j++) {
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * pi * sigma;
            Kore.at<float>(i, j) = g;
        }
    }

    return Kore;
}

vector<float> CalValueTemplate(float sigma) {
    vector<float> val_weight_arr;

    for(int i=0; i<256; i++) {
        float cur_weight = exp(-(i*i) / (2 * sigma * sigma));
        val_weight_arr.push_back(cur_weight);
    }

    return val_weight_arr;
}

Mat BilateralFilterGrey(Mat src_filtered, Mat src, int r, float sigma_s, float sigma_r) {
    Mat gaussian_kore = CalGaussianTemplate(r, sigma_s);
    vector<float> val_weight_arr = CalValueTemplate(sigma_r);

    Mat out = Mat::zeros(src.size(), src.type());

    for (int i=0; i<src.rows; i++) {
        for (int j=0; j<src.cols; j++) {
            float cur_weight = 0.0;
            float value = 0.0;
            for (int m=-r; m<=r; m++){
                for (int n=-r; n<=r; n++){
                    int cur_i = min(max(i+m, 0), src.rows-1);
                    int cur_j = min(max(j+n, 0), src.cols-1);

                    float weight = gaussian_kore.at<float>(m+r, n+r)
                            * val_weight_arr[abs(src.at<float>(i,j) - src.at<float>(cur_i, cur_j))*255];
                    value += src_filtered.at<float>(cur_i, cur_j) * weight;
                    cur_weight += weight;
                }
            }
            value = value / cur_weight;
            out.at<float>(i, j) = value;
        }
    }
    return out;
}

Mat TexturenessTransferMonoColor(Mat t_src, Mat t_ref, Mat t_hm, Mat t_detail, Mat hm_base_src, Mat detail_src) {
    int max_value = 32767;

    Mat base_src_u16 = t_src * max_value;
    Mat base_ref_u16 = t_ref * max_value;
    base_src_u16.convertTo(base_src_u16, CV_16UC1);
    base_ref_u16.convertTo(base_ref_u16, CV_16UC1);

    Mat hm_t_src = histogramMatching(base_src_u16, base_ref_u16, max_value);
    hm_t_src.convertTo(hm_t_src, CV_32FC1);
    hm_t_src = hm_t_src / max_value;

    Mat p_out = Mat::zeros(t_src.size(), CV_32FC1);
    for(int i=0; i<t_src.rows; i++) {
        for(int j=0; j<t_src.cols; j++) {
            p_out.at<float>(i, j) = fmax(0.0, (hm_t_src.at<float>(i, j) - t_hm.at<float>(i, j)) / t_detail.at<float>(i, j));
        }
    }

    Mat out = hm_base_src + p_out.mul(detail_src);

    return out;
}


Mat Textureness(Mat src, Mat ref, Mat hm_base_src, Mat detail_src, float sigma_r, float sigma_s) {
    Mat src_filtered, ref_filtered, hm_base_filtered, detail_src_filtered;
    
    GaussianBlur(src, src_filtered, Size(sigma_s, sigma_s), 0, 0);
    GaussianBlur(ref, ref_filtered, Size(sigma_s, sigma_s), 0, 0);
    GaussianBlur(hm_base_src, hm_base_filtered, Size(sigma_s, sigma_s), 0, 0);
    GaussianBlur(detail_src, detail_src_filtered, Size(sigma_s, sigma_s), 0, 0);

    src_filtered = abs(src - src_filtered);
    ref_filtered = abs(ref - ref_filtered);
    hm_base_filtered    = abs(hm_base_src - hm_base_filtered);
    detail_src_filtered = abs(detail_src - detail_src_filtered);

    int r = 9;
    Mat t_src    = BilateralFilterGrey(src_filtered, src, r, sigma_s*8, sigma_r);
    Mat t_ref    = BilateralFilterGrey(ref_filtered, ref, r, sigma_s*8, sigma_r);
    Mat t_hm     = BilateralFilterGrey(hm_base_filtered, hm_base_src, r, sigma_s*8, sigma_r);
    Mat t_detail = BilateralFilterGrey(detail_src_filtered, detail_src, r, sigma_s*8, sigma_r);

    Mat out = TexturenessTransferMonoColor(t_src, t_ref, t_hm, t_detail, hm_base_src, detail_src);

    return out;
}

float HistPrctile(Mat src, float value) {
    int max_value = 32767;
    Mat base_src_u16  = src * max_value;
    base_src_u16.convertTo(base_src_u16, CV_16UC1);

    // 计算源图像和目标图像的直方图
    cv::Mat sourceHistogram = calculateHistogram(base_src_u16, max_value);

    // 计算源图像和目标图像的累积直方图
    cv::Mat sourceCumulativeHistogram = calculateCumulativeHistogram(sourceHistogram);

    // 归一化累积直方图
    sourceCumulativeHistogram /= sourceCumulativeHistogram.at<float>(max_value-1, 0);

    float result = 0.0;
    for(int i=0; i<max_value; i++) {
        if(sourceCumulativeHistogram.at<float>(i, 0) > value) {
            result = i * 1.0f / max_value;
            break;
        }
    }
    return result;
}

float SmoothStepFunction(float x, float teta) {
    float result;

    if (x < teta) {
        result = 0.0;
    } else if (x > 2*teta) {
        result = 1.0;
    } else {
        result = 1.0 - pow(1.0-pow(x-teta, 2)/ pow(teta, 2), 2); 
    }

    return result;
}

Mat GradientReversalRemovalDetailPreservationMonoColor(Mat src, Mat textureness, float alpha) {
    Mat gxI = Mat::zeros(src.size(), CV_32FC1);
    Mat gyI = Mat::zeros(src.size(), CV_32FC1);
    Mat gxD = Mat::zeros(src.size(), CV_32FC1);
    Mat gyD = Mat::zeros(src.size(), CV_32FC1);
    Mat gx  = Mat::zeros(src.size(), CV_32FC1);
    Mat gy  = Mat::zeros(src.size(), CV_32FC1);

    for(int i=0; i<src.rows-1; i++) {
        for(int j=0; j<src.cols-1; j++) {
            gxI.at<float>(i, j) = src.at<float>(i, j+1) - src.at<float>(i, j);
            gyI.at<float>(i, j) = src.at<float>(i+1, j) - src.at<float>(i, j);
            gxD.at<float>(i, j) = textureness.at<float>(i, j+1) - textureness.at<float>(i, j);
            gyD.at<float>(i, j) = textureness.at<float>(i+1, j) - textureness.at<float>(i, j);
        }
    }

    float contrast = alpha*4;
    float beta = 1 + 3*SmoothStepFunction(contrast, 0.1);
    cout << "contrast:" << contrast << ", beta:" << beta << ", alpha:" << alpha << endl;

    for(int i=0; i<src.rows; i++) {
        float *ptr_gxD = gxD.ptr<float>(i);
        float *ptr_gyD = gyD.ptr<float>(i);
        float *ptr_gxI = gxI.ptr<float>(i);
        float *ptr_gyI = gyI.ptr<float>(i);
        float *ptr_gx  = gx.ptr<float>(i);
        float *ptr_gy  = gy.ptr<float>(i);
        for(int j=0; j<src.cols; j++) {
            if(fabs(ptr_gxD[j]) < (alpha * fabs(ptr_gxI[j]))) {
                ptr_gx[j] = alpha * ptr_gxI[j];
            } else if(fabs(ptr_gxD[j]) > fabs(beta * ptr_gxI[j])) {
                ptr_gx[j] = beta * ptr_gxI[j];
            } else {
                ptr_gx[j] = ptr_gxD[j];
            }

            if(fabs(ptr_gyD[j]) < (alpha * fabs(ptr_gyI[j]))) {
                ptr_gy[j] = alpha * ptr_gyI[j];
            } else if(fabs(ptr_gyD[j]) > fabs(beta * ptr_gyI[j])) {
                ptr_gy[j] = beta * ptr_gyI[j];
            } else {
                ptr_gy[j] = ptr_gyD[j];
            }
        }
    }

    MyPoissonFusionTest *my_poisson_test = new MyPoissonFusionTest();
    Mat out = my_poisson_test->Run(textureness, gx, gy);

    return out;
}

Mat GradientReversalRemovalDetailPreservation(Mat src, Mat textureness_dst, Mat ref) {
    int max_value = 32767;

    Mat base_text_u16 = textureness_dst * max_value;
    Mat base_ref_u16  = ref * max_value;
    base_text_u16.convertTo(base_text_u16, CV_16UC1);
    base_ref_u16.convertTo(base_ref_u16, CV_16UC1);

    Mat hm_text_src = histogramMatching(base_text_u16, base_ref_u16, max_value);
    hm_text_src.convertTo(hm_text_src, CV_32FC1);
    hm_text_src = hm_text_src / max_value;

    float prc_text_1 =  HistPrctile(hm_text_src, 0.05);
    float prc_text_2 =  HistPrctile(hm_text_src, 0.95);

    float prc_src_1 =  HistPrctile(src, 0.05);
    float prc_src_2 =  HistPrctile(src, 0.95);

    float contrast = (prc_text_2 - prc_text_1) / (prc_src_2 - prc_src_1);
    float alpha = contrast / 4.0;

    Mat out = GradientReversalRemovalDetailPreservationMonoColor(src, hm_text_src, alpha);

    return out;
}

Mat TwoScalePhotoGraphic::Run(Mat src, Mat ref) {
    int max_value = 32767;

    resize(src, src, src.size()/4*4);
    resize(ref, ref, ref.size()/4*4);

    src.convertTo(src, CV_32FC1);
    src = src / 255.0;
    ref.convertTo(ref, CV_32FC1);
    ref = ref / 255.0;

    Mat base_src, detail_src, base_ref, detail_ref;
    GetTowScaleImg(src, ref, base_src, detail_src, base_ref, detail_ref);
#if 0
    imshow("base_src", base_src);
    imshow("base_ref", base_ref);
    imshow("detail_src", abs(detail_src)*2);
    imshow("detail_ref", abs(detail_ref)*2);
#endif
    cout << "src:" << src.size() << ", ref:" << ref.size() << endl;

    detail_src = GradientReversalRemoval(src, detail_src);
    base_src = src - detail_src;

    detail_ref = GradientReversalRemoval(ref, detail_ref);
    base_ref = ref - detail_ref;

#if 0
    imshow("base_src2", base_src);
    imshow("base_ref2", base_ref);
    imshow("detail_src2", abs(detail_src)*2);
    imshow("detail_ref2", abs(detail_ref)*2);
#endif

    Mat base_src_u16 = base_src * max_value;
    Mat base_ref_u16 = base_ref * max_value;
    base_src_u16.convertTo(base_src_u16, CV_16UC1);
    base_ref_u16.convertTo(base_ref_u16, CV_16UC1);

    Mat hm_base_src = histogramMatching(base_src_u16, base_ref_u16, max_value);
    hm_base_src.convertTo(hm_base_src, CV_32FC1);
    hm_base_src = hm_base_src / max_value;
    imshow("hm_base_src", hm_base_src);

   	double minValue, maxValue;    
	minMaxLoc(src, &minValue, &maxValue, NULL, NULL);
    float sigma_r = 0.1 * (maxValue - minValue) * 255.0;
    float sigma_s = min(src.rows, src.cols) / 32 * 2 + 1;

    cout << "sigma_r:" << sigma_r << ", sigma_s:" << sigma_s << endl;
    Mat textureness_dst = Textureness(src, ref, hm_base_src, detail_src, sigma_r, sigma_s);
    textureness_dst.setTo(0.0, textureness_dst<0.0);
    textureness_dst.setTo(1.0, textureness_dst>1.0);
    imshow("textureness_dst", textureness_dst);

    Mat out = GradientReversalRemovalDetailPreservation(src, textureness_dst, ref);

    return out;
}
