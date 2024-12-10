#include "GdHdrTMO.hpp"
#include "laplace.hpp"
#include <numeric>

GdHdrTMO::GdHdrTMO() {
}

GdHdrTMO::~GdHdrTMO() {
}

vector<Mat> GdHdrTMO::BuildGaussianPy(Mat pImage) {
    Mat downImage = pImage;

    vector<Mat> pPyramid;
	while (downImage.cols > 32 && downImage.rows > 32) {
		pPyramid.push_back(downImage);
		cv::pyrDown(downImage, downImage, cv::Size(downImage.cols / 2, downImage.rows / 2));
	}

    return pPyramid;
}

void GdHdrTMO::CalculateGradient(cv::Mat& pImage, int level, cv::Mat& pGradX, cv::Mat& pGradY) {
	pGradX = cv::Mat::zeros(pImage.rows, pImage.cols, CV_32FC1);
	pGradY = cv::Mat::zeros(pImage.rows, pImage.cols, CV_32FC1);
	for (int i = 0; i < pImage.rows; i++) {
		for (int j = 0; j < pImage.cols; j++) {
			if (i == 0) {
				pGradY.at<float>(i, j) = (pImage.at<float>(i + 1, j) - pImage.at<float>(i, j)) / pow(2, level + 1);
			} else if ( i == pImage.rows - 1) {
				pGradY.at<float>(i, j) = (pImage.at<float>(i, j) - pImage.at<float>(i - 1, j)) / pow(2, level + 1);
			} else {
				pGradY.at<float>(i, j) = (pImage.at<float>(i + 1, j) - pImage.at<float>(i - 1, j)) / pow(2, level + 1);
			}

			if ( j == 0 ) {
				pGradX.at<float>(i, j) = (pImage.at<float>(i, j + 1) - pImage.at<float>(i, j)) / pow(2, level + 1);
			} else if ( j == pImage.cols - 1) {
				pGradX.at<float>(i, j) = (pImage.at<float>(i, j) - pImage.at<float>(i, j - 1)) / pow(2, level + 1);
			} else {
				pGradX.at<float>(i, j) = (pImage.at<float>(i, j + 1) - pImage.at<float>(i, j - 1)) / pow(2, level + 1);
			}
		}
    }
}

void GdHdrTMO::CalculateScaling(Mat& pGradMag, Mat& pScaling) {
	Mat temp;
	double alpha = m_alpha * mean(pGradMag)[0];

	pow((pGradMag / alpha), m_beta, temp);
	multiply(alpha / pGradMag, temp, pScaling);
}

void GdHdrTMO::CalculateAttenuations(vector<Mat> pScalings, Mat& Attenuation) {
    Mat temp;
	for (int i = pScalings.size() - 2; i >= 0; i--) {
		resize(pScalings[i + 1], temp, Size(pScalings[i].cols, pScalings[i].rows));
		multiply(pScalings[i], temp, pScalings[i]);
	}
	Attenuation = pScalings[0];
}

void GdHdrTMO::CalculateAttenuatedGradient(cv::Mat& pImage, cv::Mat& phi, cv::Mat& pGradX, cv::Mat& pGradY) {

	pGradX = cv::Mat::zeros(pImage.rows, pImage.cols, CV_32FC1);
	pGradY = cv::Mat::zeros(pImage.rows, pImage.cols, CV_32FC1);
	for (int i = 0; i < pImage.rows; i++) {
		for (int j = 0; j < pImage.cols; j++) {
			if (j + 1 >= pImage.cols) {
				pGradX.at<float>(i, j) = (pImage.at<float>(i, pImage.cols-2)-pImage.at<float>(i, j) ) * 0.5*(phi.at<float>(i, pImage.cols - 2)+phi.at<float>(i, j) );
			} else {
				pGradX.at<float>(i, j) = (pImage.at<float>(i, j + 1) - pImage.at<float>(i, j))*0.5*(phi.at<float>(i, j + 1) + phi.at<float>(i, j));
			}
			if (i + 1 >= pImage.rows) { 
				pGradY.at<float>(i, j) = (pImage.at<float>(pImage.rows-2, j) - pImage.at<float>(i, j) ) * 0.5 * (phi.at<float>(pImage.rows - 2, j) + phi.at<float>(i, j));
			} else {
				pGradY.at<float>(i, j) = (pImage.at<float>(i + 1, j) - pImage.at<float>(i, j) ) * 0.5 * (phi.at<float>(i + 1, j) + phi.at<float>(i, j));
			}
		}
    }
}

void GdHdrTMO::CalculateDivergence(cv::Mat& Gx, cv::Mat& Gy, cv::Mat& divG) {
	divG = cv::Mat::zeros(Gx.rows, Gx.cols, CV_32FC1);
	for (int i = 0; i < Gx.rows; i++) {
		for (int j = 0; j < Gx.cols; j++) {
			divG.at<float>(i, j) = Gx.at<float>(i, j) + Gy.at<float>(i, j);
			if (j > 0) { divG.at<float>(i, j) -= Gx.at<float>(i, j - 1); }
			if (i > 0) { divG.at<float>(i, j) -= Gy.at<float>(i - 1, j); }

			if (j == 0) { divG.at<float>(i, j) += Gx.at<float>(i, j); }
			if (i == 0) { divG.at<float>(i, j) += Gy.at<float>(i, j); }
		}
    }
}

Mat GdHdrTMO::ApplyToneMapping(Mat log_luma) {
    vector<Mat> gaussian_pyr = BuildGaussianPy(log_luma);
    
    Mat grad_x, grad_y, grad_mag;
    Mat p_scaling;
    vector<Mat> scaling_vector;

    for(int i=0; i<gaussian_pyr.size(); i++) {
        CalculateGradient(gaussian_pyr[i], i, grad_x, grad_y);
        magnitude(grad_x, grad_y, grad_mag);
        CalculateScaling(grad_mag, p_scaling);
        scaling_vector.push_back(p_scaling);
    }

    Mat attenuation;
	CalculateAttenuations(scaling_vector, attenuation);

    Mat attenuated_grad_x, attenuated_grad_y;
    CalculateAttenuatedGradient(log_luma, attenuation, attenuated_grad_x, attenuated_grad_y);

    Mat div_g;
	CalculateDivergence(attenuated_grad_x, attenuated_grad_y, div_g);

    return div_g;
}

void copyMatObject2Array(cv::Mat& divG, boost::multi_array<double, 2>& F){
	for (int i = 0; i < divG.rows; i++) {
		for (int j = 0; j < divG.cols; j++) {
			F[i][j] = divG.at<float>(i, j);
		}
    }
}

void copyArray2MatObject(boost::multi_array<double, 2>& U, cv::Mat& I) {
	for (int i = 0; i < I.rows; i++) {
		for (int j = 0; j < I.cols; j++) {
			I.at<float>(i, j) = U[i][j];
		}
    }
}

Mat ChangeLuminance(Mat src, Mat new_l, Mat old_l) {
    Mat out, scale_mat;
    divide(new_l, old_l, scale_mat);

    vector<Mat> channels;
    split(src, channels);
    for(int c=0; c<3; c++) {
        channels[c] = channels[c].mul(scale_mat);
    }

    merge(channels, out);

    return out;
}

Mat GdHdrTMO::FFTCalcu(Mat div_g) {
    pde::fftw_threads(4);

	double h1 = 1.0, h2 = 1.0, a1 = 1.0, a2 = 1.0;
	pde::types::boundary bdtype = pde::types::Neumann;
	double bdvalue = 0.0;
	double trunc;
    int width  = div_g.cols;
	int height = div_g.rows;

	boost::multi_array<double, 2> F(boost::extents[div_g.rows][div_g.cols]);
	boost::multi_array<double, 2> U;

	copyMatObject2Array(div_g, F);

	if (bdtype == pde::types::Neumann) 
	{
		bdvalue = pde::neumann_compat(F, a1, a2, h1, h2);
	}
	trunc = pde::poisolve(U, F, a1, a2, h1, h2, bdvalue, bdtype, false);
	pde::fftw_clean();

	/************************************
	*   Create HDR2LDR output
	*	exp(I,outlogLuma)
	*	cout = (cin / lin)^s * lour
	*************************************/

	cv::Mat I(height, width, CV_32FC1);
	cv::Mat outLuma;

	copyArray2MatObject(U, I);
	cv::exp(I, outLuma);

    double min_value, max_value;
	minMaxIdx(outLuma, &min_value, &max_value);
	outLuma.convertTo(outLuma, CV_32FC1, 1 / max_value);

    return outLuma;
}

Mat GdHdrTMO::Run(Mat src_rgb, float alpha, float beta) {
    m_alpha = alpha;
    m_beta  = beta;

    Mat src_gray;
    cvtColor(src_rgb, src_gray, COLOR_BGR2GRAY);
    
    double min_value, max_value;
    minMaxLoc(src_gray, &min_value, &max_value, NULL, NULL);
    
    Mat linhdr = src_gray / max_value;

    Mat logLuma;
	log(linhdr, logLuma);

    Mat div_g = ApplyToneMapping(logLuma);
 
    Mat outLuma = FFTCalcu(div_g);
    Mat out = ChangeLuminance(src_rgb, outLuma, src_gray);   

    return out;
}
