#include "BruceExpoBlendTMO.hpp"
#include <numeric>

BruceExpoBlendTMO::BruceExpoBlendTMO() {
}

BruceExpoBlendTMO::~BruceExpoBlendTMO() {
}

vector<Mat> BruceExpoBlendTMO::CreateLDRStackFromHDR(Mat src) {
    vector<float> light_ratio = {0.25, 0.5, 1.0, 2.0, 4.0};
    //vector<float> light_ratio = {0.25, 1.0, 4.0};
    vector<Mat> ldr_arr;

    for(int i=0; i<light_ratio.size(); i++) {
        Mat cur_src = src * light_ratio[i];
        cur_src.setTo(0, cur_src<0);     
        cur_src.setTo(1.0, cur_src>1.0);
        ldr_arr.push_back(cur_src);

        imshow(format("ldr_%d", i), cur_src);
    }

    return ldr_arr;
}

Mat entropyfilt(Mat logI) {
    //生成滤波核
    int R = 13;
    int blockSize = 2*R+1;

    uchar kernel[2*R+1][2*R+1];
    for(int i=0; i<2*R+1; i++) {
        for(int j=0; j<2*R+1; j++) {
            int v = pow(i-R, 2) + pow(j-R, 2);
            if(v<pow(R, 2)) {
                kernel[i][j] = 255;
            } else {
                kernel[i][j] = 0;
            }
        }
    }
    Mat kernel_mask = Mat(blockSize, blockSize, CV_8UC1, kernel);

    Mat entropy_mat = Mat::zeros(logI.size(), CV_32FC1);
    for(int x=R; x<logI.rows-R-1; x++) {
        for(int y=R; y<logI.cols-R-1; y++) {
            Mat block = logI(Rect(y-R, x-R, blockSize, blockSize)).clone();
            block.setTo(0, kernel_mask);
            block = block * 255;
            block.convertTo(block, CV_8UC1);

            int histSize = 256;
            float range[] = { 0,255 };
            const float* histRanges = { range };
            Mat hist_arr;
            calcHist(&block, 1, 0, Mat(), hist_arr, 1, &histSize, &histRanges, true, false);

            // 归一化直方图
            for(int i=0; i<256; i++) {
                hist_arr.at<float>(i, 0) = hist_arr.at<float>(i, 0) / (block.rows * block.cols);
            }

            // 计算熵
            float entropy = 0.0;
            for (int i=0; i<256; i++) {
                if (hist_arr.at<float>(i, 0) > 0) {
                    entropy -= hist_arr.at<float>(i, 0) * std::log2(hist_arr.at<float>(i, 0));
                }
            }
            entropy_mat.at<float>(x, y) = entropy;
        }
    }

    return entropy_mat;
}

Mat BruceExpoBlendTMO::GetResult(vector<Mat> ldr_arr, Mat src) {
    Mat totalE1 = Mat::zeros(ldr_arr[0].size(), CV_32FC1);
    vector<Mat> H_local;
    for(int i=0; i<ldr_arr.size(); i++) {
        Mat logI;
        log(ldr_arr[i]+1.0, logI);
        Mat h_mat = entropyfilt(logI);
        totalE1 = totalE1 + h_mat;
        H_local.push_back(h_mat);
        
        cout << "entropyfilt:" << i << endl;
    }

    float beb_beta = 6;
    Mat totalE2 = Mat::zeros(ldr_arr[0].size(), CV_32FC1);
    for(int i=0; i<ldr_arr.size(); i++) {
        Mat h_norm;
        divide(H_local[i], totalE1, h_norm);
        exp(beb_beta * h_norm, H_local[i]);
        totalE2 = totalE2 + H_local[i];

        cout << "totalE2:" << i << endl;
    }


    Mat out = Mat::zeros(ldr_arr[0].size(), CV_32FC3);
    for(int i=0; i<ldr_arr.size(); i++) {
        Mat h_norm;
        divide(H_local[i], totalE2, h_norm);

        vector<Mat> channels;
        split(src, channels);

        for(int c=0; c<3; c++) {
            log(channels[c]+1.0, channels[c]);
            channels[c] = channels[c].mul(h_norm);
        }

        Mat logI;
        merge(channels, logI);

        out = out + logI;
        cout << "out:" << i << endl;
    }

    exp(out, out);
    double minValue, maxValue;    // 最大值，最小值
    minMaxLoc(out, &minValue, &maxValue, NULL, NULL);

    out = (out - minValue) / (maxValue - minValue);

    return out;
}

Mat BruceExpoBlendTMO::Run(Mat src) {
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    vector<Mat> ldr_arr = CreateLDRStackFromHDR(src_gray);
    
    Mat out = GetResult(ldr_arr, src);

    return out;
}
