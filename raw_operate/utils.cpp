#include "utils.hpp"

RawUtils::RawUtils() {
}

RawUtils::~RawUtils() {
}

Mat RawUtils::Bayer2Rggb(Mat bayer){
    int height = bayer.rows;
    int width = bayer.cols;
    int chn = 4;
    Mat rggb = cv::Mat::zeros(height / 2, width / 2, CV_16UC4);

    uint16_t* data = (uint16_t*)bayer.data;
    uint16_t* res = (uint16_t*)rggb.data;
    int scale_row =  width * chn / 4;
    int scale_col = chn / 2;

    for (int row = 0; row < height; row += 2) {
        for (int col = 0; col < width; col += 2) {
            int index = row * scale_row + col * scale_col;
            res[index] = data[row * width + col];
            res[index + 1] = data[row * width + (col + 1)];
            res[index + 2] = data[(row + 1)* width + col];
            res[index + 3] = data[(row + 1)* width + (col + 1)];
        }
    }
    return rggb;
}

Mat RawUtils::Rggb2Bayer(Mat rggb){
    int height = rggb.rows;
    int width = rggb.cols;
    int chn = rggb.channels();
    rggb.convertTo(rggb, CV_16UC4);
    Mat bayer = Mat::zeros(height * 2, width * 2, CV_16UC1);

    uint16_t* data = (uint16_t*)rggb.data;
    uint16_t* res = (uint16_t*)bayer.data;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * (width * chn) + col * chn;
            res[(2 * row) * width * 2 + (2 * col)] = data[index];
            res[(2 * row) * width * 2 + (2 * col + 1)] = data[index + 1];
            res[(2 * row + 1) * width * 2 + (2 * col)] = data[index + 2];
            res[(2 * row + 1) * width * 2 + (2 * col + 1)] = data[index + 3];
        }
    }

    return bayer;
}

Mat RawUtils::Bgr2Mosaicking(Mat bgr){
    int height = bgr.rows;
    int width = bgr.cols;
    int chn = 4;
    Mat bayer = Mat::zeros(height * 2, width * 2, CV_16UC1);
    uint16_t* res = (uint16_t*)bayer.data;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * (width * chn) + col * chn;
            res[(2 * row) * width * 2 + (2 * col)]         = bgr.at<ushort>(row, col*3+0);
            res[(2 * row) * width * 2 + (2 * col + 1)]     = bgr.at<ushort>(row, col*3+1);
            res[(2 * row + 1) * width * 2 + (2 * col)]     = bgr.at<ushort>(row, col*3+1);
            res[(2 * row + 1) * width * 2 + (2 * col + 1)] = bgr.at<ushort>(row, col*3+2);
        }
    }

    return bayer;
}

Mat RawUtils::SubBlack(Mat rggb, int *black_level, int *white_level){
    std::vector<cv::Mat> channels;

    rggb.convertTo(rggb, CV_32FC4);
    split(rggb, channels);

    for(int i=0; i<4; i++) {
        channels[i] = (channels[i] - black_level[i]) / (white_level[i] - black_level[i]);
        channels[i] = channels[i] * 65535;
    }

    Mat rggb_black_sub;
    merge(channels, rggb_black_sub);

    rggb_black_sub.setTo(0, rggb_black_sub < 0);
    rggb_black_sub.convertTo(rggb_black_sub, CV_16UC4);

    return rggb_black_sub;
}

Mat RawUtils::AddBlack(Mat rggb, int *black_level){
    std::vector<cv::Mat> channels;

    rggb.convertTo(rggb, CV_32FC4);
    split(rggb, channels);

    for(int i=0; i<4; i++) {
        channels[i] = (channels[i] + black_level[i]);
    }

    Mat rggb_black_sub;
    merge(channels, rggb_black_sub);

    rggb_black_sub.convertTo(rggb_black_sub, CV_16UC4);

    return rggb_black_sub;
}

Mat RawUtils::GainAdjust(Mat rggb, float isp_gain) {
    vector<Mat> rggb_arr;
    split(rggb, rggb_arr);

    for(int ch = 0; ch < 4; ch++){
        rggb_arr[ch] *= isp_gain;
    }

    Mat rggb_gain;
    merge(rggb_arr, rggb_gain);

    return rggb_gain;
}

Mat RawUtils::AddWBgain(Mat rggb, float *wb_gain){
    vector<Mat> rggb_arr;
    split(rggb, rggb_arr);

    for(int ch = 0; ch < 4; ch++){
        rggb_arr[ch] *= wb_gain[ch];
    }

    Mat rggb_wb;
    merge(rggb_arr, rggb_wb);

    return rggb_wb;
}

Mat RawUtils::CcmAdjust(Mat rgb, float (*ccm)[3]){
    Mat ccm_rgb = Mat::zeros(rgb.size(), rgb.type());

    for(int i=0; i<rgb.rows; i++) {
        for(int j=0; j<rgb.cols; j++) {
            ccm_rgb.at<ushort>(i, j*3+0) = rgb.at<ushort>(i, j*3+0) * ccm[0][0] 
                                            + rgb.at<ushort>(i, j*3+0) * ccm[0][1]
                                            + rgb.at<ushort>(i, j*3+0) * ccm[0][2];
            ccm_rgb.at<ushort>(i, j*3+1) = rgb.at<ushort>(i, j*3+1) * ccm[0][1] 
                                            + rgb.at<ushort>(i, j*3+1) * ccm[1][1]
                                            + rgb.at<ushort>(i, j*3+1) * ccm[1][2];
            ccm_rgb.at<ushort>(i, j*3+2) = rgb.at<ushort>(i, j*3+2) * ccm[0][2] 
                                            + rgb.at<ushort>(i, j*3+2) * ccm[2][1]
                                            + rgb.at<ushort>(i, j*3+2) * ccm[2][2];
        }
    }
    return ccm_rgb;
}

Mat RawUtils::GammaAdjust(Mat ccm_rgb, float gamma){
    ccm_rgb.convertTo(ccm_rgb, CV_32FC3);
    ccm_rgb = ccm_rgb / 65535;
    Mat gamma_rgb = Mat::zeros(ccm_rgb.size(), ccm_rgb.type());

    for(int i=0; i<ccm_rgb.rows; i++) {
        for(int j=0; j<ccm_rgb.cols; j++) {
            gamma_rgb.at<float>(i, j*3+0) = pow(ccm_rgb.at<float>(i, j*3+0), gamma);
            gamma_rgb.at<float>(i, j*3+1) = pow(ccm_rgb.at<float>(i, j*3+1), gamma);
            gamma_rgb.at<float>(i, j*3+2) = pow(ccm_rgb.at<float>(i, j*3+2), gamma);
        }
    }
     
    gamma_rgb = gamma_rgb * 65535;
    gamma_rgb.convertTo(gamma_rgb, CV_16UC3);

    return gamma_rgb;
}


Mat RawUtils::RawRead(string file_path, int height, int width, int type) {
    ifstream fin;
    fin.open(file_path,  ios::binary);
    if(!fin) {
        std::cerr << "open failed: " << file_path << std::endl;
    }

    // seek函数会把标记移动到输入流的结尾
    fin.seekg(0, fin.end);
    // tell会告知整个输入流（从开头到标记）的字节数量
    int length = fin.tellg();
    // 再把标记移动到流的开始位置
    fin.seekg(0, fin.beg);
    cout << "file length: " << length << endl;

    // load buffer
    char* buffer = new char [length];
    // read函数读取（拷贝）流中的length各字节到buffer
    fin.read(buffer, length);

    // construct opencv mat and show image
    Mat image(Size(height, width), type, buffer);

    return image;
}
