#include "LaplacianPyramid.h"

int GetLevelCount(int rows, int cols, int desired_base_size) {
    int min_dim = std::min(rows, cols);

    double log2_dim = std::log2(min_dim);
    double log2_des = std::log2(desired_base_size);

    int count = ceil(abs(log2_dim - log2_des));

    return count;
}

vector<int> GetLevelSize(int level, vector<int> subwindow) {
    for (int i = 0; i < level; i++) {
        subwindow[0] = (subwindow[0]>>1) + (subwindow[0]%2);
        subwindow[1] = subwindow[1]>>1;
        subwindow[2] = (subwindow[2]>>1) + (subwindow[2]%2);
        subwindow[3] = subwindow[3]>>1;
    }

    return subwindow;
}

float WeightingFunction(int i, float a) {
    switch (i) {
        case 0: return a;
        case -1: case 1: return 0.25;
        case -2: case 2: return 0.25 - 0.5 * a;
    }
    return 0;
}

Mat PopulateTopLevel(Mat previous, int kRows, int kCols, int row_offset, int col_offset) {
    float kA = 0.4;
    Mat cur_mat = Mat::zeros(Size(kCols, kRows), CV_32FC1);

    int kEndRow = row_offset + 2 * cur_mat.rows;
    int kEndCol = col_offset + 2 * cur_mat.cols;
    for (int y = row_offset; y < kEndRow; y += 2) {
        for (int x = col_offset; x < kEndCol; x += 2) {
            float value = 0;
            float total_weight = 0;

            int row_start = max(0, y - 2);
            int row_end = min(previous.rows - 1, y + 2);
            for (int n = row_start; n <= row_end; n++) {
                float row_weight = WeightingFunction(n - y, kA);

                int col_start = std::max(0, x - 2);
                int col_end = std::min(previous.cols - 1, x + 2);
                for (int m = col_start; m <= col_end; m++) {
                    float weight = row_weight * WeightingFunction(m - x, kA);
                    total_weight += weight;
                    value += weight * previous.at<float>(n, m);
                }
            }
            cur_mat.at<float>(y >> 1, x >> 1) = value / total_weight;
        }
    }
    return cur_mat;
}

Mat Expand(Mat input, int out_rows, int out_cols, int row_start, int row_end, int col_start, int col_end) {
    float kA = 0.4;
    Mat output = Mat::zeros(out_rows, out_cols, input.type());
    Mat upsamp = Mat::zeros(out_rows, out_cols, input.type());
    Mat norm   = Mat::zeros(out_rows, out_cols, CV_32F);

    int row_offset = ((row_start % 2) == 0) ? 0 : 1;
    int col_offset = ((col_start % 2) == 0) ? 0 : 1;

    for (int i=row_offset; i <out_rows; i+=2) {
        for (int j=col_offset; j<out_cols; j+=2) {
            upsamp.at<float>(i, j) = input.at<float>(i >> 1, j >> 1);
            norm.at<float>(i, j) = 1;
        }
    }

    Mat filter(5, 5, CV_32F);
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            filter.at<float>(i + 2, j + 2) = WeightingFunction(i, kA) * WeightingFunction(j, kA);
        }
    }

    for (int i = 0; i < output.rows; i++) {
        int row_start = max(0, i - 2);
        int row_end   = min(output.rows - 1, i + 2);
        for (int j = 0; j < output.cols; j++) {
            int col_start = max(0, j - 2);
            int col_end   = min(output.cols - 1, j + 2);

            float value = 0;
            float total_weight = 0;
            for (int n = row_start; n <= row_end; n++) {
                for (int m = col_start; m <= col_end; m++) {
                    float weight = filter.at<float>(n - i + 2, m - j + 2);
                    value += weight * upsamp.at<float>(n, m);
                    total_weight += weight * norm.at<float>(n, m);
                }
            }
            output.at<float>(i, j) = value / total_weight;
        }
    }
    return output;
}

vector<Mat> GaussianPyramid(Mat img, int level, int row_start, int row_end, int col_start, int col_end) {
    vector<Mat> pyr;
    pyr.push_back(img);
    vector<int> subwindow = {row_start, row_end, col_start, col_end};

    Mat item;
    for (int i=0; i<level; i++) {
        item = pyr.back();
        vector<int> prev_subwindow, current_subwindow;
        prev_subwindow = GetLevelSize(pyr.size() - 1, subwindow);
        current_subwindow = GetLevelSize(pyr.size(), subwindow);
        int kRows = current_subwindow[1] - current_subwindow[0] + 1;
        int kCols = current_subwindow[3] - current_subwindow[2] + 1;
        int row_offset = ((prev_subwindow[0] % 2) == 0) ? 0 : 1;
        int col_offset = ((prev_subwindow[2] % 2) == 0) ? 0 : 1;
 
        Mat cur_mat = PopulateTopLevel(pyr[pyr.size()-1], kRows, kCols, row_offset, col_offset);
        pyr.push_back(cur_mat);
    }

    return pyr;
}

vector<Mat> LaplacianPyramid(Mat img, int level, int row_start, int row_end, int col_start, int col_end) {
    vector<Mat> gauss_pyr = GaussianPyramid(img, level, row_start, row_end, col_start, col_end);
    vector<Mat> lap_pyr;
    
    for (int i=0; i<level; i++) {
        int out_rows = gauss_pyr[i].rows;
        int out_cols = gauss_pyr[i].cols;
        if(i>0) {
            row_start = (row_start / 2) + (row_start%2);
            row_end = row_end / 2;
            col_start = (col_start / 2) + (col_start%2);
            col_end = col_end / 2;
        }

        Mat up_mat = Expand(gauss_pyr[i+1], out_rows, out_cols, row_start, row_end, col_start, col_end);
        lap_pyr.push_back(gauss_pyr[i] - up_mat);
    }
    lap_pyr.push_back(gauss_pyr[level]);
    return lap_pyr;
}

Mat LapReconstruct(vector<Mat> output) {
    Mat out;
    for (int i=(int)(output.size()-2); i>=0; i--) {
        int out_rows = output[i].rows;
        int out_cols = output[i].cols;
        if(out.empty()){
            out = Expand(output[i+1], out_rows, out_cols, 0, output[i].rows-1, 0, output[i].cols-1);
        } else {
            out = Expand(out, out_rows, out_cols, 0, output[i].rows-1, 0, output[i].cols-1);
        }
        out = out + output[i];
    }
    return out;
}

