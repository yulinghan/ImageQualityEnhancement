#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>
#include "RemappingFunction.h"
#include "LaplacianPyramid.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    float kAlpha = 0.3;
    float kBeta = 2.3;
    float kSigmaR = 0.1;

    Mat src = imread(argv[1], 0);
    src.convertTo(src, CV_32FC1, 1/255.0);
 
    int desired_base_size = 30;
    int level = GetLevelCount(src.rows, src.cols, desired_base_size);
	level = level -1;

    vector<Mat> gaussian_src = GaussianPyramid(src, level, 0, src.rows-1, 0, src.cols-1);

    vector<Mat> output;
    for(int i=0; i<(int)gaussian_src.size()-1; i++) {
        Mat cur_mat = Mat::zeros(gaussian_src[i].size(), CV_32FC1);
        output.push_back(cur_mat);
    }
    output.push_back(gaussian_src[gaussian_src.size()-1].clone());

    for(int k=0; k<level; k++) {
        int subregion_size = 3*((1<<(k+2))-1);
        int subregion_r = subregion_size/2;

        for (int m=0; m<gaussian_src[k].rows; m++) {
            int full_res_y = (1<<k)*m;
            int roi_y0 = full_res_y-subregion_r;
            int roi_y1 = full_res_y+subregion_r+1;
            Range row_range(max(0, roi_y0), min(roi_y1, src.rows));
            int full_res_roi_y = full_res_y - row_range.start;

            for(int n=0; n<gaussian_src[k].cols; n++) {
                int full_res_x = (1<<k)*n;
                int roi_x0 = full_res_x-subregion_r;
                int roi_x1 = full_res_x+subregion_r+1;
                Range col_range(max(0, roi_x0), min(roi_x1, src.cols));
                int full_res_roi_x = full_res_x - col_range.start;
                Mat r0 = src(row_range, col_range);
                Mat remapped =  Evaluate(r0, gaussian_src[k].at<float>(m, n), kAlpha, kBeta, kSigmaR);
                vector<Mat> remapped_src = LaplacianPyramid(remapped, k+1, row_range.start, row_range.end-1, col_range.start, col_range.end-1);
                output[k].at<float>(m, n) = remapped_src[k].at<float>(full_res_roi_y >> k, full_res_roi_x >> k);
            }
        }
    }

    Mat out = LapReconstruct(output);
	out = out * 255;
	out.convertTo(out, CV_8UC1);
    imwrite(argv[2], out);

    return 0;
}
