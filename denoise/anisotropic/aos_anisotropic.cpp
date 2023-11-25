#include "aos_anisotropic.hpp"

MyAosAnisotropicTest::MyAosAnisotropicTest() {
}

MyAosAnisotropicTest::~MyAosAnisotropicTest() {
}

Mat MyAosAnisotropicTest::pm_g1(Mat Lx, Mat Ly, float k) {
	Mat weight;
	exp(-(Lx.mul(Lx) + Ly.mul(Ly))/(k*k), weight);

    return weight;
}

Mat MyAosAnisotropicTest::pm_g2(Mat Lx, Mat Ly, float k) {
	Mat weight = 1. / (1. + (Lx.mul(Lx)+Ly.mul(Ly)) / (k*k));

    return weight;
}

Mat MyAosAnisotropicTest::pm_g3(Mat Lx, Mat Ly, float k) {
	Mat modg;
    pow((Lx.mul(Lx) + Ly.mul(Ly))/(k*k),4,modg);

	Mat weight;
    exp(-3.315/modg, weight);
    weight = 1.0 - weight;

    return weight;
}

float MyAosAnisotropicTest::Compute_K_Percentile(Mat Lx, Mat Ly, int nbins) {
    float hist[nbins];
	for(unsigned int i = 0; i < nbins; i++ ) {
		hist[i] = 0.0;
	}
 
	//找到图像最大梯度
	float lx, ly, modg;
	float hmax = 0.0;
	for(int i=1; i < Lx.rows-1; i++) {
		for(int j=1; j<Lx.cols-1; j++) {
            lx = *(Lx.ptr<float>(i)+j);
            ly = *(Ly.ptr<float>(i)+j);
			modg = sqrt(lx*lx + ly*ly);
	
			if( modg > hmax) {
				hmax = modg;
			}
		}
	}
 
	//图像梯度归一化，并根据输入参数nbins，设置梯度直方图数量，生成梯度直方图
    float npoints = 0.0;
	int nbin = 0;
	for(int i=1; i<Lx.rows-1; i++ ) {
		for(int j=1; j<Lx.cols-1; j++ ) {
            lx = *(Lx.ptr<float>(i)+j);
            ly = *(Ly.ptr<float>(i)+j);
			modg = sqrt(lx*lx + ly*ly);
 
			if(modg != 0.0) {
				nbin = floor(nbins*(modg/hmax));
                if( nbin == nbins ) {
                    nbin--;
                }
 
				hist[nbin]++;
				npoints++;
			}
		}
	}
	
	//统计梯度直方图，占梯度70%时候的域值
	int nthreshold = (int)(npoints*0.7);
	int k, nelements;
	for(k = 0, nelements = 0; nelements < nthreshold && k < nbins; k++) {
		nelements = nelements + hist[k];
	}
	
	//生成控制扩散级别的对比度因子kperc；
	float kperc = hmax*((float)(k)/(float)nbins);	
	
	return kperc;
}

Mat MyAosAnisotropicTest::Thomas(Mat a, Mat b, Mat Ld) {
	int n = a.rows;
	Mat m = Mat::zeros(a.rows,a.cols,CV_32F);
	Mat l = Mat::zeros(b.rows,b.cols,CV_32F);
	Mat y = Mat::zeros(Ld.rows,Ld.cols,CV_32F);
	Mat x = Mat::zeros(a.rows,a.cols,CV_32F);

	for(int j=0; j<m.cols; j++) {
		*(m.ptr<float>(0)+j) = *(a.ptr<float>(0)+j);
	}

	for(int j=0; j<y.cols; j++) {
		*(y.ptr<float>(0)+j) = *(Ld.ptr<float>(0)+j);
	}

	for(int k=1; k<n; k++) {
		for(int j=0; j < l.cols; j++) {
			*(l.ptr<float>(k-1)+j) = *(b.ptr<float>(k-1)+j) / *(m.ptr<float>(k-1)+j);
		}

		for(int j=0; j<m.cols; j++) {
			*(m.ptr<float>(k)+j) = *(a.ptr<float>(k)+j) - *(l.ptr<float>(k-1)+j)*(*(b.ptr<float>(k-1)+j));
		}

		for(int j=0; j<y.cols; j++) {
			*(y.ptr<float>(k)+j) = *(Ld.ptr<float>(k)+j) - *(l.ptr<float>(k-1)+j)*(*(y.ptr<float>(k-1)+j));
		}
	}

	for(int j=0; j<y.cols; j++) {
		*(x.ptr<float>(n-1)+j) = (*(y.ptr<float>(n-1)+j))/(*(m.ptr<float>(n-1)+j));
	}

	for(int i=n-2; i>=0; i--) {
		for(int j=0; j<x.cols; j++) {
			*(x.ptr<float>(i)+j) = (*(y.ptr<float>(i)+j) - (*(b.ptr<float>(i)+j))*(*(x.ptr<float>(i+1)+j)))/(*(m.ptr<float>(i)+j));
		}
	}

	return x;
}

Mat MyAosAnisotropicTest::AosRows(Mat Ldprev, Mat c, float stepsize) {
	Mat qr = Mat::zeros(c.size(), CV_32FC1);
	Mat py = Mat::zeros(c.size(), CV_32FC1);

	for(int i=0; i<qr.rows; i++) {
		for(int j = 0; j < qr.cols; j++) {
			*(qr.ptr<float>(i)+j) = *(c.ptr<float>(i)+j) + *(c.ptr<float>(i+1)+j);
		}
	}

	for( int j=0; j<py.cols; j++) {
		*(py.ptr<float>(0)+j) = *(qr.ptr<float>(0)+j);
	}

	for( int j=0; j<py.cols; j++) {
		*(py.ptr<float>(py.rows-1)+j) = *(qr.ptr<float>(qr.rows-1)+j);
	}

	for( int i = 1; i < py.rows-1; i++) {
		for( int j = 0; j < py.cols; j++) {
			*(py.ptr<float>(i)+j) = *(qr.ptr<float>(i-1)+j) + *(qr.ptr<float>(i)+j);
		}
	}

	Mat ay = 1.0 + stepsize*py;
	Mat by = -stepsize*qr;

	Mat Lty = Thomas(ay, by, Ldprev);

	return Lty;
}

Mat MyAosAnisotropicTest::AosColumns(Mat Ldprev, Mat c, float stepsize) {
	Mat qc = Mat::zeros(c.size(), CV_32FC1);
	Mat px = Mat::zeros(c.size(), CV_32FC1);

	for(int j=0; j<qc.cols; j++) {
		for(int i=0; i<qc.rows; i++) {
			*(qc.ptr<float>(i)+j) = *(c.ptr<float>(i)+j) + *(c.ptr<float>(i)+j+1);
		}
	}

	for(int i=0; i<px.rows; i++) {
		*(px.ptr<float>(i)) = *(qc.ptr<float>(i));
	}

	for( int i=0; i<px.rows; i++) {
		*(px.ptr<float>(i)+px.cols-1) = *(qc.ptr<float>(i)+qc.cols-1);
	}

	for( int j=1; j<px.cols-1; j++) {
		for( int i=0; i<px.rows; i++) {
			*(px.ptr<float>(i)+j) = *(qc.ptr<float>(i)+j-1) + *(qc.ptr<float>(i)+j);
		}
	}

	Mat ax = 1.0 + stepsize*px.t();
	Mat bx = -stepsize*qc.t();

	Mat Ltx = Thomas(ax, bx, Ldprev.t());

	return Ltx;
}

Mat MyAosAnisotropicTest::AosStepScalar(Mat Ldprev, Mat c, float stepsize) {
	Mat Lty = AosRows(Ldprev,c,stepsize);
	Mat Ltx = AosColumns(Ldprev,c,stepsize);

	Mat dst = 0.5*(Lty + Ltx.t());

	return dst;
}

Mat MyAosAnisotropicTest::Run(Mat src) {
	src.convertTo(src, CV_32FC1, 1./255.0);

	//计算图像x和y方向梯度
    int ksize_x = 7, ksize_y = 7;
    float gscale = 3.0;
    Mat gaussian, Lx, Ly;
    GaussianBlur(src, gaussian, Size(ksize_x,ksize_y), gscale, gscale);

    float k1[]={1, -1}, k2[3][1]={1, -1};
    Mat Kore1 = Mat(1, 2, CV_32FC1,k1);
    Mat Kore2 = Mat(2, 1, CV_32FC1,k2);
    Point point1(-1, 0);
    Point point2(0, -1);
    filter2D(gaussian, Lx, -1, Kore1, point1, 0, BORDER_CONSTANT);
    filter2D(gaussian, Ly, -1, Kore2, point2, 0, BORDER_CONSTANT);

	float k = Compute_K_Percentile(Lx, Ly, 128);
	cout << "k:" << k << endl;

	Mat weight = pm_g3(Lx, Ly, k);

	Mat temp_mat = AosStepScalar(src, weight, 15.0);
	temp_mat = temp_mat * 255;
	temp_mat.convertTo(temp_mat, CV_8UC1);	

	return temp_mat;
}
