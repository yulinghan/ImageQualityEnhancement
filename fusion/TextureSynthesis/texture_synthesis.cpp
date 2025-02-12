#include "texture_synthesis.hpp"

MyTextureSynthesisTest::MyTextureSynthesisTest() {
}

MyTextureSynthesisTest::~MyTextureSynthesisTest() {
}

void Shift(Mat &src) {
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    Mat q0(src, Rect(0, 0, cx, cy));   // ROI区域的左上
    Mat q1(src, Rect(cx, 0, cx, cy));  // ROI区域的右上
    Mat q2(src, Rect(0, cy, cx, cy));  // ROI区域的左下
    Mat q3(src, Rect(cx, cy, cx, cy)); // ROI区域的右下

    //交换象限（左上与右下进行交换）
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    //交换象限（右上与左下进行交换）
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

Mat myfft(Mat &src) {
    Mat src2 = Mat::zeros(src.size(), CV_32F);
    Mat planes[] = {src, src2};

    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);

    return complexI;
}

Mat myifft(Mat src) {
    Mat re;
    idft(src, re);
    re = re / re.rows / re.cols;
    Mat re_channels[2];
    split(re, re_channels);

    Mat result = re_channels[0];

    return result;
}

Mat GetNoiseMat(Mat src, int max_value) {
    Mat noiseImage = Mat::zeros(src.size(), CV_16UC1);

    for (int y=0; y<src.rows; y++) {
        for (int x=0; x<src.cols; x++) {
            int randomValue = rand() % max_value;
            noiseImage.at<ushort>(y, x) = static_cast<ushort>(randomValue);
        }
    }
    return noiseImage;
}

// 计算图像的直方图
cv::Mat calculateHistogram(const cv::Mat& image, int max_value) {
    cv::Mat histogram;
    int histSize = max_value; // 直方图的bin数量
    float range[] = {0, max_value}; // 像素值范围
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
    sourceCumulativeHistogram /= sourceCumulativeHistogram.at<float>(max_value-1, 0);
    targetCumulativeHistogram /= targetCumulativeHistogram.at<float>(max_value-1, 0);

    // 创建查找表
    cv::Mat lookupTable(1, max_value, CV_16U);
    for (int i = 0; i < max_value; i++) {
        // 使用二分查找找到最接近的匹配值
        auto it = lower_bound(targetCumulativeHistogram.ptr<float>(0), targetCumulativeHistogram.ptr<float>(max_value) + 1, sourceCumulativeHistogram.at<float>(i, 0));
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

void GetFFTFreq(Mat complexI, vector<Mat> &freq_arr) {
    vector<Mat> mask_freq_arr;
    for(int i=0; i<8; i++) {
        Mat mask_freq = Mat::zeros(complexI.size(), CV_32FC1);
        mask_freq_arr.push_back(mask_freq);
    }

    Mat tmp_mask0 = Mat::zeros(complexI.size(), CV_32FC1);
    Mat tmp_mask1 = Mat::zeros(complexI.size(), CV_32FC1);
    for(int i=0; i<tmp_mask0.rows; i++) {
        for(int j=0; j<tmp_mask0.cols; j++) {
            if(i>j) {
                tmp_mask0.at<float>(i, j) = 1.0;
            }
            if(i>tmp_mask0.cols-j) {
                tmp_mask1.at<float>(i, j) = 1.0;
            }
        }
    }

    for(int i=0; i<complexI.rows; i++) {
        for(int j=0; j<complexI.cols; j++) {
            if(tmp_mask0.at<float>(i, j) == 1.0 && tmp_mask1.at<float>(i, j) == 1.0) {
                if(j<complexI.cols/2) {
                    mask_freq_arr[0].at<float>(i, j) = 1.0;
                } else {
                    mask_freq_arr[4].at<float>(i, j) = 1.0;
                }
            } else if(tmp_mask0.at<float>(i, j) == 1.0 && tmp_mask1.at<float>(i, j) == 0.0) {
                if(i<complexI.rows/2) {
                    mask_freq_arr[1].at<float>(i, j) = 1.0;
                } else {
                    mask_freq_arr[5].at<float>(i, j) = 1.0;
                }
            } else if(tmp_mask0.at<float>(i, j) == 0.0 && tmp_mask1.at<float>(i, j) == 1.0) {
                if(i<complexI.rows/2) {
                    mask_freq_arr[2].at<float>(i, j) = 1.0;
                } else {
                    mask_freq_arr[6].at<float>(i, j) = 1.0;
                }
            } else {
                if(j<complexI.cols/2) {
                    mask_freq_arr[3].at<float>(i, j) = 1.0;
                } else {
                    mask_freq_arr[7].at<float>(i, j) = 1.0;
                }
            }
        }
    }

    Mat all_freq_mask;
    for(int i=0; i<8; i++) {
        GaussianBlur(mask_freq_arr[i], mask_freq_arr[i], Size(57, 57), 0, 0);
        if(i == 0) {
            all_freq_mask = mask_freq_arr[i].clone();
        } else {
            all_freq_mask += mask_freq_arr[i];
        }
    }

    for(int i=0; i<8; i++) {
        divide(mask_freq_arr[i], all_freq_mask, mask_freq_arr[i]);
    }

    vector<Mat> planes;
    for(int i=0; i<8; i++) {
        split(complexI.clone(), planes);
        Shift(planes[0]);
        Shift(planes[1]);
        planes[0] = planes[0].mul(mask_freq_arr[i]);
        planes[1] = planes[1].mul(mask_freq_arr[i]);
        Shift(planes[0]);
        Shift(planes[1]);

        Mat cur_h;
        merge(planes, cur_h);
        freq_arr.push_back(cur_h);
    }
}

vector<vector<Mat>> MyTextureSynthesisTest::GetHaarPyr(Mat src, int level) {
    vector<vector<Mat>>  haar_pyr_arr;
    for(int i=0; i<level; i++) {
        vector<Mat> haar_arr;
        Mat LL, HH, tmp_small, tmp_big;

        resize(src, tmp_small, src.size()/2, 0, 0);
        GaussianBlur(tmp_small, LL, Size(3, 3), 0, 0);

        resize(LL, tmp_big, src.size(), 0, 0);
        HH = src - tmp_big;

        Mat complexI = myfft(HH);

        vector<Mat> freq_arr;
        GetFFTFreq(complexI, freq_arr);

        haar_arr.push_back(LL);
        for(int k=0; k<freq_arr.size(); k++) {
            freq_arr[k] = myifft(freq_arr[k]);
            haar_arr.push_back(freq_arr[k]);
        }
        haar_pyr_arr.push_back(haar_arr);

        src = LL.clone();
    }

    return haar_pyr_arr;
}

Mat MyTextureSynthesisTest::GetTextureSynthesis(vector<vector<Mat>> src_haar_pyr_arr, vector<vector<Mat>> noise_haar_pyr_arr, int max_value) {
    int level = src_haar_pyr_arr.size();
    Mat out   = noise_haar_pyr_arr[level-1][0];

    for(int i=level-1; i>=0; i--) {
        resize(out, out, noise_haar_pyr_arr[i][1].size());
        for(int k=1; k<src_haar_pyr_arr[i].size(); k++) {
            src_haar_pyr_arr[i][k]   = src_haar_pyr_arr[i][k]*max_value + max_value / 2;
            src_haar_pyr_arr[i][k].setTo(0, src_haar_pyr_arr[i][k]<0);
            src_haar_pyr_arr[i][k].setTo(max_value-1, src_haar_pyr_arr[i][k]>=max_value);
            src_haar_pyr_arr[i][k].convertTo(src_haar_pyr_arr[i][k], CV_16UC1);

            noise_haar_pyr_arr[i][k] = noise_haar_pyr_arr[i][k]*max_value + max_value / 2;
            noise_haar_pyr_arr[i][k].setTo(0, noise_haar_pyr_arr[i][k]<0);
            noise_haar_pyr_arr[i][k].setTo(max_value-1, noise_haar_pyr_arr[i][k]>=max_value);
            noise_haar_pyr_arr[i][k].convertTo(noise_haar_pyr_arr[i][k], CV_16UC1);

            noise_haar_pyr_arr[i][k] = histogramMatching(noise_haar_pyr_arr[i][k], src_haar_pyr_arr[i][k], max_value);
            noise_haar_pyr_arr[i][k].convertTo(noise_haar_pyr_arr[i][k], CV_32FC1);
            noise_haar_pyr_arr[i][k] = noise_haar_pyr_arr[i][k] - max_value / 2;
            noise_haar_pyr_arr[i][k] = noise_haar_pyr_arr[i][k] / max_value;
            
            out = out + noise_haar_pyr_arr[i][k];
        }
    }

    return out;
}

Mat MyTextureSynthesisTest::Run(Mat src) {
    int level = 5;
    int max_value = 32768;

    src.convertTo(src, CV_16UC1);
    src = src * (max_value / 256);

    Mat noise_mat = GetNoiseMat(src, max_value);
    imshow("noise_mat", noise_mat);
  
    Mat hist_noise_mat = histogramMatching(noise_mat, src, max_value);

    src.convertTo(src, CV_32FC1);
    hist_noise_mat.convertTo(hist_noise_mat, CV_32FC1);
    src = src / max_value;
    hist_noise_mat = hist_noise_mat / max_value;

    Mat texture_synt_out = hist_noise_mat;
    for(int i=0; i<10; i++) {
        vector<vector<Mat>> src_haar_pyr_arr   = GetHaarPyr(src, level);
        vector<vector<Mat>> noise_haar_pyr_arr = GetHaarPyr(texture_synt_out, level);
        texture_synt_out = GetTextureSynthesis(src_haar_pyr_arr, noise_haar_pyr_arr, max_value);
        imshow("texture_synt_out", texture_synt_out);
        waitKey(0);
    }

    return texture_synt_out;
}
