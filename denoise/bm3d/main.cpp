#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string.h>
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

#include "bm3d.h"

#define DCT       4
#define BIOR      5

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    //! Variables initialization
    const bool useSD_1 = true;
    const bool useSD_2 = true;
    const int patch_size = 8;
    float fSigma = 30;

    //! Declarations
    vector<float> img_noisy, img_basic, img_denoised;
    unsigned width, height;

    //! Load image
    Mat src = imread(argv[1], 0);
    height = src.rows;
    width  = src.cols;

	for(int i=0; i<src.rows; i++) {
		for(int j=0; j<src.cols; j++) {
			img_noisy.push_back(src.at<uchar>(i, j));
		}
	}

    //! Denoising
    if (run_bm3d(fSigma, img_noisy, img_basic, img_denoised, width, height,
                useSD_1, useSD_2, patch_size) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    cout << endl << "Save images...";

    Mat basic_mat = Mat::zeros(src.size(), CV_8UC1);
    Mat denoised_mat = Mat::zeros(src.size(), CV_8UC1);

    int mm = 0;
	for(int i=0; i<src.rows; i++) {
		for(int j=0; j<src.cols; j++) {
			basic_mat.at<uchar>(i, j) = img_basic[mm];
			denoised_mat.at<uchar>(i, j) = img_denoised[mm];
			mm += 1;
		}
	}

	imwrite(argv[2], basic_mat);
    imwrite(argv[3], denoised_mat);

    cout << "done." << endl;

    return EXIT_SUCCESS;
}
