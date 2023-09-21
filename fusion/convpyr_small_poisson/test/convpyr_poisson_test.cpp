#include "convpyr_poisson_test.hpp"

MyConvPyrPoissonTest::MyConvPyrPoissonTest() {
}

MyConvPyrPoissonTest::~MyConvPyrPoissonTest() {
}

Mat MyConvPyrPoissonTest::CalImageDivAndGrad(Mat src1, Mat src2, Mat mask1, Mat mask2) {
    float k1[]={1, -1, 0}, k2[3][1]={1, -1, 0};
    Mat Kore1 = Mat(1, 3, CV_32FC1,k1);
    Mat Kore2 = Mat(3, 1, CV_32FC1,k2);
    Point point1(-1, 0);
    Point point2(0, -1);

    Mat src1_x, src1_y, src2_x, src2_y, src3_x, src3_y, input_x, input_y;
    filter2D(src1, src1_x, -1, Kore1, point1, 0, BORDER_CONSTANT); 
    filter2D(src1, src1_y, -1, Kore2, point2, 0, BORDER_CONSTANT);
    filter2D(src2, src2_x, -1, Kore1, point1, 0, BORDER_CONSTANT); 
    filter2D(src2, src2_y, -1, Kore2, point2, 0, BORDER_CONSTANT);

    Mat fusion_x = src2_x.clone();
    Mat fusion_y = src2_y.clone();

	for(int i=0; i<mask1.rows; i++) {
		for(int j=0; j<mask1.cols; j++) {
			if(mask1.at<float>(i, j)>0.5){
				float value_x=0, value_y=0, weight=0;
				value_x = src1_x.at<float>(i, j) * mask1.at<float>(i, j) + src2_x.at<float>(i, j) * mask2.at<float>(i, j);
				value_y = src1_y.at<float>(i, j) * mask1.at<float>(i, j) + src2_y.at<float>(i, j) * mask2.at<float>(i, j);
				weight  = 0.00001 + mask1.at<float>(i, j) + mask2.at<float>(i, j);

				fusion_x.at<float>(i, j) = value_x / weight;
				fusion_y.at<float>(i, j) = value_y / weight;
			}
		}
	}

    float div_k1[]={0, 1, -1}, div_k2[3][1]={0, 1, -1};
    Kore1 = Mat(1, 3, CV_32FC1, div_k1);
    Kore2 = Mat(3, 1, CV_32FC1, div_k2);

    Mat div_x, div_y;
    filter2D(fusion_x, div_x, -1, Kore1, point1, 0, BORDER_CONSTANT);
    filter2D(fusion_y, div_y, -1, Kore2, point2, 0, BORDER_CONSTANT);

    Mat div_g = div_x + div_y;
    return div_g;
}
 
Mat MyConvPyrPoissonTest::evalf(Mat cur_div_old, Mat h1, Mat h2, Mat g){
	Mat cur_div = -cur_div_old;
	int maxLevel = (int)(log2(max(cur_div.rows , cur_div.cols)));
	Point point(-1, -1);

	vector<Mat> pyr_arr;
	int fs = h1.rows;
	copyMakeBorder(cur_div, cur_div, fs, fs, fs, fs, BORDER_CONSTANT);
	pyr_arr.push_back(cur_div);
	for(int i=1; i<=maxLevel; i++) {
		Mat down_mat;
		filter2D(pyr_arr[i-1], down_mat, -1, h1, point, 0, BORDER_CONSTANT);
		resize(down_mat, down_mat, down_mat.size()/2, 0, 0, INTER_NEAREST);
		copyMakeBorder(down_mat, down_mat, fs, fs, fs, fs, BORDER_CONSTANT);
		pyr_arr.push_back(down_mat);
	}

	vector<Mat> f_pyr_arr;
	Mat out;
	filter2D(pyr_arr[maxLevel], out, -1, g, point, 0, BORDER_CONSTANT);
	f_pyr_arr.push_back(out);
	for(int i=maxLevel-1; i>=0; i--) {
		Mat rd = f_pyr_arr[maxLevel- i - 1];
		rd = rd(Rect(fs, fs, rd.cols-fs*2, rd.rows-fs*2));
		Mat new_rd = Mat::zeros(pyr_arr[i].size(), CV_32FC1);
		for(int m=0; m<new_rd.rows; m+=2) {
			for(int n=0; n<new_rd.cols; n+=2) {
				new_rd.at<float>(m, n) = rd.at<float>(m/2, n/2);
			}
		}
		Mat f1, f2;
		filter2D(new_rd, f1, -1, h2, point, 0, BORDER_CONSTANT);
		filter2D(pyr_arr[i], f2, -1, g, point, 0, BORDER_CONSTANT);

		f_pyr_arr.push_back(f1 + f2);

	}
	maxLevel = f_pyr_arr.size();
	out = f_pyr_arr[maxLevel-1];
	out = out(Rect(fs, fs, out.cols-fs*2, out.rows-fs*2));

	return out;
}

void MyConvPyrPoissonTest::constructKernels(Mat w_img, Mat &h1, Mat &h2, Mat &g) {
    h1 = Mat::zeros(Size(5, 1), CV_32FC1);
    g  = Mat::zeros(Size(3, 1), CV_32FC1);

    h1.at<float>(0, 0) = w_img.at<float>(0, 0);
    h1.at<float>(0, 1) = w_img.at<float>(0, 1);
    h1.at<float>(0, 2) = w_img.at<float>(0, 2);
    h1.at<float>(0, 3) = h1.at<float>(0, 1);
    h1.at<float>(0, 4) = h1.at<float>(0, 0);

    h1 = h1.t() * h1;
    h2 = h1;

    g.at<float>(0, 0) = w_img.at<float>(0, 3);
    g.at<float>(0, 1) = w_img.at<float>(0, 4);
    g.at<float>(0, 2) = w_img.at<float>(0, 3);

    g = g.t() * g;
}              

Mat MyConvPyrPoissonTest::Run(Mat src1, Mat src2, Mat mask) {
	Mat mask1 = mask;
    Mat mask2 = 255 - mask1;
    mask1.convertTo(mask1, CV_32FC1, 1/255.0);
    mask2.convertTo(mask2, CV_32FC1, 1/255.0);

    src1.convertTo(src1, CV_32FC1, 1/255.0);
    src2.convertTo(src2, CV_32FC1, 1/255.0);

    int fs = 1;
    copyMakeBorder(src1, src1, fs, fs, fs, fs, BORDER_CONSTANT);
    copyMakeBorder(src2, src2, fs, fs, fs, fs, BORDER_CONSTANT);
    copyMakeBorder(mask1, mask1, fs, fs, fs, fs, BORDER_CONSTANT);
    copyMakeBorder(mask2, mask2, fs, fs, fs, fs, BORDER_CONSTANT);

    Mat w_img = Mat::zeros(Size(5, 1),CV_32FC1);
    randu(w_img, Scalar::all(0.0), Scalar::all(1.0));
    //0.166143, 0.500589, 0.668934, 0.184746, 0.550890
    w_img.at<float>(0, 0) = -0.1820;
    w_img.at<float>(0, 1) = -0.5007;
    w_img.at<float>(0, 2) = -0.6373;
    w_img.at<float>(0, 3) = 0.1767;
    w_img.at<float>(0, 4) = 0.5589;

    Mat h1, h2, g;
    constructKernels(w_img, h1, h2, g);

    vector<Mat> channels1, channels2;
    split(src1, channels1);
    split(src2, channels2);

    for(int i=0; i<(int)channels1.size(); i++) {
        Mat cur_div = CalImageDivAndGrad(channels1[i], channels2[i], mask1, mask2);
        channels1[i] = evalf(cur_div, h1, h2, g);
        channels1[i] = channels1[i](Rect(fs, fs, channels1[i].cols-fs*2, channels1[i].rows-fs*2));
    }

    Mat out;
    merge(channels1, out);

	return out;
}
