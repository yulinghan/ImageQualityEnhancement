#include "pull_push_nlm.hpp"

MyPullPushNlmTest::MyPullPushNlmTest() {
}

MyPullPushNlmTest::~MyPullPushNlmTest() {
}

float MyPullPushNlmTest::MseBlock(Mat m1, Mat m2) {
    float sum = 0.0;
    for (int j = 0; j < m1.rows; j++) {
        uchar *data1 = m1.ptr<uchar>(j);
        uchar *data2 = m2.ptr<uchar>(j);
        for (int i = 0; i < m1.cols; i++) {
            sum += (data1[i] - data2[i]) * (data1[i] - data2[i]);
        }
    }
    sum = sum / (m1.rows*m2.cols);
    return sum;
}

Mat MyPullPushNlmTest::DownFuse(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    Mat dst   = Mat::zeros(src.size()/2, CV_8UC1);
    Mat cur_p = Mat::zeros(src.size()/2, CV_32FC1);
    int boardSize = halfKernelSize + halfSearchSize;
    Mat boardSrc;
    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展

    float h2 = h*h;
    int rows = src.rows;
    int cols = src.cols;

    for (int j = boardSize; j < boardSize + rows - 1; j+=2) {
        uchar *dst_out = dst.ptr<uchar>((j - boardSize)/2);
        float *dst_p = cur_p.ptr<float>((j - boardSize)/2);
        for (int i = boardSize; i < boardSize + cols - 1; i+=2) {
            Mat patchA = boardSrc(Range(j - halfKernelSize, j + halfKernelSize + 1), Range(i - halfKernelSize, i + halfKernelSize + 1));
            float w = 0;
            float value = 0;
            float sumw1 = 0;
            float sumw2 = 0;
            float p = 0;
            float w_arr1[halfSearchSize*2+1][halfSearchSize*2+1];
            float w_arr2[halfSearchSize*2+1][halfSearchSize*2+1];

            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {
                    Mat patchB = boardSrc(Range(j + sr - halfKernelSize, j + sr + halfKernelSize + 1), Range(i + sc - halfKernelSize, i + sc + halfKernelSize + 1));
                    float d2 = MseBlock(patchA, patchB);

                    w = exp(-d2 / h2);
                    w_arr1[sr+halfSearchSize][sc+halfSearchSize] = w;
                    sumw1 += w;

                    if(pull_weight_p_arr.size()>0) {
                        float p_weight = pull_weight_p_arr[pull_weight_p_arr.size()-1].at<float>(j - boardSize, i - boardSize);
                        w = w * p_weight;
                    }

                    w_arr2[sr+halfSearchSize][sc+halfSearchSize] = w;
                    sumw2 += w;
                }
            }

            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                uchar *boardSrc_p = boardSrc.ptr<uchar>(j + sr);
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {
                    p += pow(w_arr1[sr+halfSearchSize][sc+halfSearchSize]/sumw1, 2);
                    w_arr2[sr+halfSearchSize][sc+halfSearchSize] = w_arr2[sr+halfSearchSize][sc+halfSearchSize] / sumw2;
                    value += boardSrc_p[i + sc] * w_arr2[sr+halfSearchSize][sc+halfSearchSize];
                }
            }
            p = 1.0 / p;
            dst_p[(i - boardSize)/2] = p;

            dst_out[(i - boardSize)/2] = fmin(fmax(value, 0.0), 255.0);
        }
    }

    pull_weight_p_arr.push_back(cur_p);

    return dst;
}

Mat MyPullPushNlmTest::UpFuse(Mat src_f, Mat src_c, float h, int halfKernelSize, int halfSearchSize) {
    Mat dst      = Mat::zeros(src_f.size(), CV_8UC1);
    Mat weight_p = Mat::zeros(src_f.size(), CV_32FC1);
    int boardSize = halfKernelSize + halfSearchSize;
    Mat boardSrc_f, boardSrc_c;
    copyMakeBorder(src_f, boardSrc_f, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展
    copyMakeBorder(src_c, boardSrc_c, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展

    Mat Gaussian_src_c;
    GaussianBlur(boardSrc_c, Gaussian_src_c, Size(3, 3), 0, 0);

    float h2 = h*h;
    int rows = src_f.rows;
    int cols = src_f.cols;

    for (int j = boardSize; j < boardSize + rows - 2; j++) {
        uchar *dst_out = dst.ptr<uchar>(j - boardSize);
        float *p_ptr   = weight_p.ptr<float>(j - boardSize);
        for (int i = boardSize; i < boardSize + cols - 2; i++) {
            Mat patchA   = boardSrc_f(Range(j-halfKernelSize, j+halfKernelSize+1), Range(i-halfKernelSize, i+halfKernelSize+1));

            int newj = (j-boardSize)/2 + boardSize;
            int newi = (i-boardSize)/2 + boardSize;
            Mat patchA_c = Gaussian_src_c(Range(newj-halfKernelSize, newj+halfKernelSize+1), Range(newi-halfKernelSize, newi+halfKernelSize+1));
            float w = 0;
            float sumw = 0;
            float sumw2 = 0;

            float w_arr1[halfSearchSize*2+1][halfSearchSize*2+1];
            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {
                    Mat patchB = boardSrc_f(Range(j + sr - halfKernelSize, j + sr + halfKernelSize + 1), Range(i + sc - halfKernelSize, i + sc + halfKernelSize+1));
                    float d2 = MseBlock(patchA, patchB);

                    w = exp(-d2 / h2);
                    w_arr1[sr+halfSearchSize][sc+halfSearchSize] = w;
                    sumw += w;
                }
            }

            float w_arr2[halfSearchSize*2+1][halfSearchSize*2+1];
            float w_arr3[halfSearchSize*2+1][halfSearchSize*2+1];
            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {
                    Mat patchB_c = Gaussian_src_c(Range(newj+sr-halfKernelSize, newj+sr+halfKernelSize+1), Range(newi+sc-halfKernelSize, newi+sc+halfKernelSize+1));
                    float d2 = MseBlock(patchA_c, patchB_c);
                    w = exp(-d2 / h2);
                    w_arr3[sr+halfSearchSize][sc+halfSearchSize] = w;
                    sumw2 += w;

                    if(push_weight_p_arr.size()>0) {
                        float p_value = push_weight_p_arr[push_weight_p_arr.size()-1].at<float>((j-boardSize)/2, (i-boardSize)/2);
//                        p_value = fmin(p_value-3, 0.0);
                        w = w * p_value;
                    }
                    w_arr2[sr+halfSearchSize][sc+halfSearchSize] = w;
                    sumw  += w;
                }
            }

            float value = 0;
            float p_value = 0;
            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++) {
                uchar *src_f_ptr = boardSrc_f.ptr(j + sr);
                uchar *src_c_ptr = boardSrc_c.ptr(newj + sr);

                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++) {
                    p_value += pow(w_arr3[sr+halfSearchSize][sc+halfSearchSize] / sumw2, 2);
                    w_arr1[sr+halfSearchSize][sc+halfSearchSize] = w_arr1[sr+halfSearchSize][sc+halfSearchSize] / sumw;
                    w_arr2[sr+halfSearchSize][sc+halfSearchSize] = w_arr2[sr+halfSearchSize][sc+halfSearchSize] / sumw;
                    value += src_f_ptr[i + sc]   * w_arr1[sr+halfSearchSize][sc+halfSearchSize];
                    value += src_c_ptr[newi + sc] * w_arr2[sr+halfSearchSize][sc+halfSearchSize];
                }
            }
            dst_out[i - boardSize] = fmax(fmin(value, 255.0), 0.0);
            p_ptr[i - boardSize] = 1.0/p_value;
        }
    }

    push_weight_p_arr.push_back(weight_p);

    return dst;
}

void MyPullPushNlmTest::PullNlm(Mat src, float h, int halfKernelSize, int halfSearchSize) {
    int num_layers = 4;
    float cur_h1[4] = {5.7, 4.5, 3.8, 3.5};
    float cur_h2[4] = {1.0, 1.0, 1.0, 1.0};
    pull_src_arr.clear();

    resize(src, src, (src.size()/2)*2);
    pull_src_arr.push_back(src);
    for(int i=0; i<num_layers; i++) {
        Mat cur_mat = DownFuse(pull_src_arr[i], cur_h1[i], halfKernelSize, halfSearchSize);
        resize(cur_mat, cur_mat, (cur_mat.size()/2)*2);
        pull_src_arr.push_back(cur_mat);

        Mat tmp;
        resize(cur_mat, tmp, src.size(), 0, 0, INTER_NEAREST);
        imshow(format("old_mat_n_%d", i), tmp);
    }

    int level = pull_src_arr.size();
    Mat y_mat = pull_src_arr[level-1].clone();
    for(int i=level-2; i>=0; i--) {
        y_mat = UpFuse(pull_src_arr[i], y_mat, cur_h2[i], halfKernelSize, halfSearchSize);
        Mat tmp;
        resize(y_mat, tmp, src.size(), 0, 0, INTER_NEAREST);
        imshow(format("new_mat_n_%d", i), tmp);
    }
    imwrite("y_mat_2.png", y_mat);
}
