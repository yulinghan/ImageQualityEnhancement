#include "haar_wave.hpp"

MyHaarWaveTest::MyHaarWaveTest() {
}

MyHaarWaveTest::~MyHaarWaveTest() {
}

Mat MyHaarWaveTest::harr_wave_recover(Mat src_img, int level) {
	Mat image = src_img.clone();
	src_img.convertTo(image, CV_32FC1);
	Mat dst = Mat::zeros(image.size(), CV_32FC1);
	for (int k = 0; k < level; ++k) {
		int row = image.rows;
		int col = image.cols;
		Rect rect(0, 0, col / pow(2, level - k - 1), row / pow(2, level - k - 1));
		Mat image1 = image(rect).clone();
		int current_rows = image1.rows;
		int current_cols = image1.cols;
		int half_row = current_rows / 2;
		int half_col = current_cols / 2;
		Mat temp_img1(current_rows, current_cols, CV_32FC1);
		int temp_int = 0;
		for (int i = 0; i < half_row; ++i) {
			float *pup = image1.ptr<float>(i);
			float *pdown = image1.ptr<float>(i + half_row);
			temp_int = i * 2;
			float *p1 = temp_img1.ptr<float>(temp_int);
			float *p2 = temp_img1.ptr<float>(temp_int +1);
			for (int j = 0; j < current_cols; ++j) {
				p1[j] = (pup[j] + pdown[j]);
				p2[j] = (pup[j] - pdown[j]);
			}
		}

		Mat image2(image1.size(), CV_32FC1);
		for (int i = 0; i < current_rows; ++i) {
			float *psrc = temp_img1.ptr<float>(i);
			float *ptmp = image2.ptr<float>(i);
			for (int j = 0; j < half_col; ++j) {
				temp_int = j << 1;
				ptmp[temp_int] = psrc[j] + psrc[j + half_col];
				ptmp[temp_int+1] = psrc[j] - psrc[j + half_col];
			}
		}
		image2.copyTo(image(rect));
	}
	image.convertTo(dst, CV_8UC1);
    
    return dst;
}

Mat MyHaarWaveTest::harr_wave_denoise(Mat src, int threshold, int level) {
    int low_rows = src.rows / pow(2, level);
    int low_cols = src.cols / pow(2, level);
    for(int i=0; i<src.rows; i++) {
        for(int j=0; j<src.cols; j++) {
            if(i>low_rows || j>low_cols) {
                if(fabs(src.at<float>(i, j)) < threshold) {
                    src.at<float>(i, j) = 0.0;
                }
            }
        }
    }
    return src;
}

Mat MyHaarWaveTest::harr_wave_decompose(Mat src, int level) {
    Mat image;
	src.convertTo(image, CV_32FC1);
	Mat dst = Mat::zeros(image.size(), CV_32FC1);
	for (int k = 0; k < level; ++k) {
		int row = image.rows;
		int col = image.cols;
		Mat image1(row, col, CV_32FC1, Scalar::all(0));
		Mat image2(row, col, CV_32FC1,Scalar::all(0));
		int half_row = row / 2;
		int half_col = col / 2;
		
		for (int i = 0; i < row; ++i) {
			float *psrc = image.ptr<float>(i);
			float *ptmp = image1.ptr<float>(i);
			for (int j = 0; j < half_col; ++j) {
				int a = j << 1;
				ptmp[j] = (psrc[a] + psrc[a + 1]) * 0.5;
				ptmp[j + half_col] = (psrc[a] - psrc[a + 1]) * 0.5;
			}
		}
		
		for (int i = 0; i < half_row; ++i) {
			float *pcurrent = image1.ptr<float>(2 * i);
			float *pnext = image1.ptr<float>(2 * i + 1);

			float *p1 = image2.ptr<float>(i);
			float *p2 = image2.ptr<float>(i + half_row);

			for (int j = 0; j < col; ++j) {
				p1[j] = (pcurrent[j] + pnext[j])*0.5;
				p2[j] = (pcurrent[j] - pnext[j])*0.5;
			}
		}

		image = image2(Rect(0, 0, image2.cols / 2, image2.rows / 2));
		image2.copyTo(dst(Rect(0, 0, image2.cols, image2.rows)));
	}

	return dst;
}
