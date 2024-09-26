#include "direction.hpp"

MyDirectionTest::MyDirectionTest() {
}

MyDirectionTest::~MyDirectionTest() {
}

short MyDirectionTest::CalcKernel(uchar *data, char *kernel, int number) {
    short sum = 0.0;
    for(int i = 0; i < 25; i++) {
        sum += data[i] * kernel[i];
    }

    sum = sum / number;
    return sum;
}

Mat MyDirectionTest::GetDirectionEdge(Mat src) {
	Mat out = Mat::zeros(src.size(), CV_8UC1);

	vector<Mat> edge_arr;
	for(int i=0; i<4; i++) {
		Mat cur_mat = Mat::zeros(src.size(), CV_8UC1);
		edge_arr.push_back(cur_mat);
	}

	for(int i=0; i<src.rows; i++) {
		uchar *ptr_src_l2 = src.ptr(max(i - 2, 0));
		uchar *ptr_src_l1 = src.ptr(max(i - 1, 0));
		uchar *ptr_src   = src.ptr(i);
		uchar *ptr_src_n1 = src.ptr(min(i + 1, src.rows - 1));
		uchar *ptr_src_n2 = src.ptr(min(i + 2, src.rows - 1));
		for(int j=0; j<src.cols; j++) {
			uchar data[25] = {
				ptr_src_l2[max(j-2, 0)], ptr_src_l2[max(j-1, 0)], ptr_src_l2[j], ptr_src_l2[min(j+1, src.cols-1)], ptr_src_l2[min(j+2, src.cols-1)],
				ptr_src_l1[max(j-2, 0)], ptr_src_l1[max(j-1, 0)], ptr_src_l1[j], ptr_src_l1[min(j+1, src.cols-1)], ptr_src_l1[min(j+2, src.cols-1)],
				ptr_src[max(j-2, 0)],    ptr_src[max(j-1, 0)],    ptr_src[j],    ptr_src[min(j+1, src.cols-1)],    ptr_src[min(j+2, src.cols-1)],
				ptr_src_n1[max(j-2, 0)], ptr_src_n1[max(j-1, 0)], ptr_src_n1[j], ptr_src_n1[min(j+1, src.cols-1)], ptr_src_n1[min(j+2, src.cols-1)],
				ptr_src_n2[max(j-2, 0)], ptr_src_n2[max(j-1, 0)], ptr_src_n2[j], ptr_src_n2[min(j+1, src.cols-1)], ptr_src_n2[min(j+2, src.cols-1)],
			};
			for(int k=0; k<4; k++) {
				short value = CalcKernel(data, kernel_kirsch[k], 26);
                edge_arr[k].at<uchar>(i, j) = abs(value);
			}
		}
	}

    for(int i=0; i<4; i++) {
        out += edge_arr[i]/4;
    }

	return out;
}

Mat MyDirectionTest::GetAdjustMat(Mat src, Mat edge_mat, int r, float scale) {
    int edge_max = 50;
    float edge_weight_arr[50] = {
        0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0
    };
    Mat out = Mat::zeros(src.size(), src.type());

    Mat gauss_mat;
    GaussianBlur(src, gauss_mat, Size(r, r), 0, 0);

    edge_mat.setTo(edge_max, edge_mat>edge_max);

    for(int i=0; i<gauss_mat.rows; i++) {
        uchar *ptr_src   = src.ptr(i);
        uchar *ptr_gauss = gauss_mat.ptr(i);
        uchar *ptr_edge  = edge_mat.ptr(i);
        uchar *ptr_out   = out.ptr(i);
        for(int j=0; j<gauss_mat.cols; j++) {
            int value = ptr_src[j] - ptr_gauss[j];
            value = value * edge_weight_arr[ptr_edge[j]] * scale;
            ptr_out[j] = max(min(value + ptr_src[j], 255), 0);
        }
    }

	return out;
}

Mat MyDirectionTest::Run(Mat src, int r, float scale) {
	Mat edge_mat = GetDirectionEdge(src);
	Mat out = GetAdjustMat(src, edge_mat, r, scale);

	return out;
}
