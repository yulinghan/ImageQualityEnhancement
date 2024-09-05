#include "BanterleTMO.hpp"

BanterleTMO::BanterleTMO() {
}

BanterleTMO::~BanterleTMO() {
}

float BanterleTMO::sign(float x) {
    if (x > 0) {
        return 1;
    } else if (x < 0) {
        return -1;
    } else {
        return 0;
    }
}

void BanterleTMO::region_grow(Mat imgBin, int x, int y, int cur_value, vector<int> &addr_array_x, vector<int> &addr_array_y) {
    int addrx, addry;
    for(int i=x-1; i<x+2; i++) {
        for(int j=y-1; j<y+2; j++) {
            addrx = min(max(i, 0), imgBin.rows-1);
            addry = min(max(j, 0), imgBin.cols-1);
            if(imgBin.at<uchar>(addrx, addry) == cur_value) {
                addr_array_x.push_back(addrx);
                addr_array_y.push_back(addry);
                imgBin.at<uchar>(addrx,addry) = 0;
            }
        }
    }
}

Mat BanterleTMO::seg_area(Mat imgBin_ori, int &seg_num) {
    Mat imgBin = imgBin_ori.clone();

    Mat out = Mat::zeros(imgBin.size(), CV_8UC1);
    vector<int> addr_array_x, addr_array_y;
    int k = 0;

    for(int i=0; i<out.rows; i++) {
        for(int j=0; j<out.cols; j++) {
            if(imgBin.at<uchar>(i,j) > 0) {
                addr_array_x.clear();
                addr_array_y.clear();
                addr_array_x.push_back(i);
                addr_array_y.push_back(j);

                int cur_value = imgBin.at<uchar>(i,j);
                imgBin.at<uchar>(i,j) = 0;
                k = 0;
                while(k < (int)addr_array_x.size()) {
                    region_grow(imgBin, addr_array_x[k], addr_array_y[k], cur_value, addr_array_x, addr_array_y);
                    out.at<uchar>(addr_array_x[k], addr_array_y[k]) = seg_num;
                    k += 1;
                }
                seg_num += 1;
            }
        }
    }
    return out;
}

vector<int> BanterleTMO::GetSegAreaNum(Mat seg_mat, int seg_num) {
    vector<int> seg_pixel_num;
    seg_pixel_num.resize(seg_num);

    for(int i=0; i<seg_mat.rows; i++) {
        for(int j=0; j<seg_mat.cols; j++) {
            seg_pixel_num[seg_mat.at<uchar>(i, j)] +=1;
        }
    }

    return seg_pixel_num;
}

Mat BanterleTMO::GetHdrSeg(Mat src_blur) {
    double minValue, maxValue;    // 最大值，最小值
    cv::minMaxLoc(src_blur, &minValue, &maxValue, NULL, NULL);
    cout << "最大值：" << maxValue <<"最小值："<<minValue<<std::endl;

    float l10Min = log10(minValue);
    float l10Max = log10(maxValue);
    cout << "1 l10Min:" << l10Min << ", l10Max:" << l10Max << endl;

    float sMin = sign(l10Min);
    float sMax = sign(l10Max);
    cout << "sMin:" << sMin << ", sMax:" << sMax << endl;

    if(sMin > 0) {
        l10Min = sMin * floor(abs(l10Min));
    } else {
        l10Min = sMin * ceil(abs(l10Min));
    }

    if(sMax > 0) {
        l10Max = sMax * ceil(abs(l10Max));
    } else {
        l10Max = sMax * floor(abs(l10Max));
    }
    cout << "2 l10Min:" << l10Min << ", l10Max:" << l10Max << endl;

    Mat imgBin = Mat::zeros(src_blur.size(), CV_8UC1);
    float nLevels = l10Max - l10Min + 1;
    cout << "nLevels:" << nLevels << endl;

    for(int i=l10Min; i<l10Max; i++) {
        float bMin = pow(10, i);
        float bMax = pow(10, i + 1);

        Mat cur_mask = Mat::zeros(src_blur.size(), CV_8UC1);
        cur_mask.setTo(255, src_blur >= bMin);
        cur_mask.setTo(0,   src_blur >= bMax);

        imgBin.setTo(i - l10Min + 1, cur_mask);
    }

    return imgBin;
}
    
vector<vector<int>> BanterleTMO::GetSegNeighbor(Mat seg_mat, int seg_num) {
    vector<vector<int>> seg_neighbor_arr2;
    for(int i=0; i<seg_num; i++) {
        vector<int> seg_neighbor_arr;
        seg_neighbor_arr.resize(seg_num);
        seg_neighbor_arr2.push_back(seg_neighbor_arr);
    }

    for(int k=0; k<seg_num; k++) {
        for(int i=1; i<seg_mat.rows-1; i++) {
            for(int j=1; j<seg_mat.cols-1; j++) {
                if(seg_mat.at<uchar>(i, j) == k) {
                    if(seg_mat.at<uchar>(i, j-1) != k) {
                        seg_neighbor_arr2[k][seg_mat.at<uchar>(i, j-1)] = 1;
                    }
                    if(seg_mat.at<uchar>(i, j+1) != k) {
                        seg_neighbor_arr2[k][seg_mat.at<uchar>(i, j+1)] = 1;
                    } 
                    if(seg_mat.at<uchar>(i-1, j) != k) {
                        seg_neighbor_arr2[k][seg_mat.at<uchar>(i-1, j)] = 1;
                    } 
                    if(seg_mat.at<uchar>(i+1, j) != k) {
                        seg_neighbor_arr2[k][seg_mat.at<uchar>(i+1, j)] = 1;
                    } 
                }
            }
        }
    }
    return seg_neighbor_arr2;
}

Mat BanterleTMO::GetSegMerge(Mat seg_mat, int seg_num, float thres, vector<int> seg_pixel_num, vector<vector<int>> seg_neighbor_arr2) {
    float thres_num = seg_mat.rows * seg_mat.cols * thres;

    for(int i=0; i<seg_num; i++) {
        if(seg_pixel_num[i]<thres_num) {
            int max_neigh = 0;
            for(int j=0; j<seg_num; j++) {
                if(seg_neighbor_arr2[i][j]>0 && seg_pixel_num[max_neigh]<seg_pixel_num[j]) {
                    max_neigh = j;
                }
            }

            seg_mat.setTo(max_neigh, seg_mat == i);
        }
    }

    return seg_mat;
}

vector<int> BanterleTMO::GetHdrToMergeNum(Mat seg_mat, Mat imgBin, int seg_num) {
    vector<int> hdrseg_merge_arr;
    hdrseg_merge_arr.resize(seg_num);

    for(int i=0; i<seg_mat.rows; i++) {
        for(int j=0; j<seg_mat.rows; j++) {
            hdrseg_merge_arr[seg_mat.at<uchar>(i, j)] = imgBin.at<uchar>(i, j);
        }
    }

    return hdrseg_merge_arr;
}

Mat BanterleTMO::Run(Mat src) {
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat src_blur;
    GaussianBlur(src_gray, src_blur, Size(3, 3), 0, 0);

    Mat imgBin = GetHdrSeg(src_blur);
    imshow("imgBin", imgBin*50);

    int seg_num = 0;
    Mat seg_mat = seg_area(imgBin, seg_num);
    imshow("seg_mat", seg_mat*10);
    cout << "seg_num:" << seg_num <<  endl;

    vector<int> hdrseg_merge_arr = GetHdrToMergeNum(seg_mat, imgBin, seg_num);

    vector<int> seg_pixel_num = GetSegAreaNum(seg_mat, seg_num);
    vector<vector<int>> seg_neighbor_arr2 = GetSegNeighbor(seg_mat, seg_num);

    float thres = 0.005;
    Mat seg_merge = GetSegMerge(seg_mat, seg_num, thres, seg_pixel_num, seg_neighbor_arr2);
    imshow("seg_merge", seg_merge*50);

    Mat out = Mat::zeros(seg_merge.size(), CV_8UC1);
    for(int i=0; i<hdrseg_merge_arr.size(); i++) {
        out.setTo(hdrseg_merge_arr[i], seg_merge==i);
    }
    
    return out;
}
