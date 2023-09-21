#include "convpyr_poisson_train.hpp"

MyConvPyrPoissonTrain::MyConvPyrPoissonTrain() {
}

MyConvPyrPoissonTrain::~MyConvPyrPoissonTrain() {
}

void MyConvPyrPoissonTrain::CalImageDivAndGrad(Mat src, Mat &div_g) {                                                                                                                                                             
    float k1[]={1, -1, 0}, k2[3][1]={1, -1, 0};
    Mat Kore1 = Mat(1, 3, CV_32FC1,k1);
    Mat Kore2 = Mat(3, 1, CV_32FC1,k2);
    Point point1(-1, 0);
    Point point2(0, -1);
        
    Mat src_x, src_y;
    filter2D(src, src_x, -1, Kore1, point1, 0, BORDER_CONSTANT);
    filter2D(src, src_y, -1, Kore2, point2, 0, BORDER_CONSTANT);
    
    float div_k1[]={0, 1, -1}, div_k2[3][1]={0, 1, -1};
    Kore1 = Mat(1, 3, CV_32FC1, div_k1);
    Kore2 = Mat(3, 1, CV_32FC1, div_k2);
    
    Mat div_x, div_y;
    filter2D(src_x, div_x, -1, Kore1, point1, 0, BORDER_CONSTANT);
    filter2D(src_y, div_y, -1, Kore2, point2, 0, BORDER_CONSTANT);

    div_g = div_x + div_y;
}

void MyConvPyrPoissonTrain::constructKernels(Mat w_img, Mat &h1, Mat &h2, Mat &g) {                                                                                                                                               
    h1 = Mat::zeros(Size(kernel_size_, 1), CV_32FC1);
    g  = Mat::zeros(Size(kernel_size_ - 2, 1), CV_32FC1);

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

Mat MyConvPyrPoissonTrain::evalf(Mat cur_div_old, Mat h1, Mat h2, Mat g){
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

double objective(const gsl_vector *x, void *params) {
	MyConvPyrPoissonTrain *cur_train = (MyConvPyrPoissonTrain *)params;
	int kernel_size = cur_train->kernel_size_;
	Mat cur_i   = cur_train->cur_i_;
	Mat cur_div = cur_train->cur_div_;

    Mat w_img = Mat::zeros(Size(kernel_size, 1), CV_32FC1);
	for(int i=0; i<kernel_size; i++) {
	    w_img.at<float>(0, i) = gsl_vector_get(x, i);
	}
    
    Mat h1, h2, g;
    cur_train->constructKernels(w_img, h1, h2, g);
    Mat res = cur_train->evalf(cur_div, h1, h2, g);

	res = res(Rect(kernel_size, kernel_size, res.cols-kernel_size, res.rows-kernel_size));
	Mat cur_i2 = cur_i(Rect(kernel_size, kernel_size, cur_i.cols-kernel_size, cur_i.rows-kernel_size));

    Mat error_d = abs(cur_i2 - res);
    Scalar neam = mean(error_d);

    return neam[0];
}

void MyConvPyrPoissonTrain::Run(Mat src) {
	Mat w_img = Mat::zeros(Size(5, 1),CV_32FC1);
    cv::randu(w_img, Scalar::all(0.0), Scalar::all(1.0));

	int h = src.rows;
    int w = src.cols;
    int n_sc_ref = int(log(min(h, w)) / log(2));
    int n_scales = 1;
    while(n_scales < n_sc_ref) {
        n_scales++;
    }
	printf("all iter:%d\n", n_scales);

    for(int i=3; i<=n_scales; i++) {
        int dim = pow(2, i);
        resize(src, cur_i_, Size(dim, dim), 0, 0, INTER_NEAREST);

        int fs = 1;
        copyMakeBorder(cur_i_, cur_i_, fs, fs, fs, fs, BORDER_CONSTANT);
        CalImageDivAndGrad(cur_i_, cur_div_);

        size_t n = kernel_size_; // dimension of the problem
        gsl_vector *x = gsl_vector_alloc(n); // initial guess
        for(int m=0; m<n; m++) {
            gsl_vector_set(x, m, w_img.at<float>(0, m));
        }

        gsl_vector *ss = gsl_vector_alloc(n); // initial guess
        gsl_vector_set_all(ss, 0.01);

        gsl_multimin_function obj_func = {objective, n, this};
        gsl_multimin_fminimizer *minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, n);
        gsl_multimin_fminimizer_set(minimizer, &obj_func, x, ss);
		int status;
        int iter = 0;
        do {
            iter++;
            status = gsl_multimin_fminimizer_iterate(minimizer);
            if (status) {
                break;
            }
            double size = gsl_multimin_fminimizer_size(minimizer);
            status = gsl_multimin_test_size(size, 1e-8);
        } while (status == GSL_CONTINUE && iter < 1000);

        printf("cur iter %d:Found minimum at (", i);
        for (int i = 0; i < n; ++i) {
            w_img.at<float>(0, i) = gsl_vector_get(minimizer->x, i);
            printf("%f", w_img.at<float>(0, i));
            if (i < n - 1) {
                printf(", ");
            }
        }
        printf(") = %g\n", minimizer->fval);
    }
}
