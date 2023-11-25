#include "anisotropic.hpp"

MyAnisotropicTest::MyAnisotropicTest() {
}

MyAnisotropicTest::~MyAnisotropicTest() {
}

float MyAnisotropicTest::pm_g1(float value, float k) {
    float weight = exp(-value*value/(k*k));

    return weight;
}

float MyAnisotropicTest::pm_g2(float value, float k) {
    float weight = 1.0 / (1.0 + (value*value)/(k*k));

    return weight;
}

float MyAnisotropicTest::pm_g3(float value, float k) {
    float weight;
    if(value*value == 0) {
        weight = 1;
    } else {
        float modg = pow(value/k, 8);
        weight = 1.0 - exp(-3.315 / modg);
    }
   
    return weight;
}

float MyAnisotropicTest::Compute_K_Percentile(Mat img, int nbins) {
    float hist[nbins];
	for(unsigned int i = 0; i < nbins; i++ ) {
		hist[i] = 0.0;
	}
 
	//计算图像x和y方向梯度
	int ksize_x = 7, ksize_y = 7;
	float gscale = 3.0;
	Mat gaussian, Lx, Ly;
	GaussianBlur(img, gaussian, Size(ksize_x,ksize_y), gscale, gscale);
			
	float k1[]={1, -1}, k2[3][1]={1, -1};
    Mat Kore1 = Mat(1, 2, CV_32FC1,k1);
    Mat Kore2 = Mat(2, 1, CV_32FC1,k2);
    Point point1(-1, 0);
    Point point2(0, -1);
    filter2D(gaussian, Lx, -1, Kore1, point1, 0, BORDER_CONSTANT);
    filter2D(gaussian, Ly, -1, Kore2, point2, 0, BORDER_CONSTANT);

	//找到图像最大梯度
	float lx, ly, modg;
	float hmax = 0.0;
	for( int i = 1; i < gaussian.rows-1; i++) {
		for( int j = 1; j < gaussian.cols-1; j++) {
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
	for( int i = 1; i < gaussian.rows-1; i++ ) {
		for( int j = 1; j < gaussian.cols-1; j++ ) {
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

Mat MyAnisotropicTest::Run(Mat src) {
	src.convertTo(src, CV_32FC1, 1./255.0);

    Mat anisotropic_mat;
    Mat temp_mat = src.clone();

    int iterations=30;
    float lambda=0.15;
	float k = Compute_K_Percentile(src, 128);
	cout << "k:" << k << endl;

    for(int t = 0;t < iterations; ++t) {
        anisotropic_mat = temp_mat.clone();
        for(int i=1; i<src.rows-1; i++) {
            float *data_ori_ptr  = anisotropic_mat.ptr<float>(i);
            float *data_prev_ptr = anisotropic_mat.ptr<float>(i-1);
            float *data_next_ptr = anisotropic_mat.ptr<float>(i+1);
            float *temp_mat_ptr  = temp_mat.ptr<float>(i);

            for(int j=1; j<src.cols-1; j++) {
                float up    = data_prev_ptr[j]  - data_ori_ptr[j];
                float down  = data_next_ptr[j]  - data_ori_ptr[j];
                float left  = data_ori_ptr[j-1] - data_ori_ptr[j];
                float right = data_ori_ptr[j+1] - data_ori_ptr[j];

                // 根据散度计算传导系数
                float up_coefficient    = pm_g3(up, k);
                float down_coefficient  = pm_g3(down, k);
                float left_coefficient  = pm_g3(left, k);
                float right_coefficient = pm_g3(right, k);

                float value = temp_mat_ptr[j] + lambda*(up*up_coefficient
                            + down*down_coefficient
                            + left*left_coefficient
                            + right*right_coefficient);
                temp_mat_ptr[j] = fmin(fmax(value, 0.0), 1.0);
            }
        }
    }

	temp_mat = temp_mat * 255.0;
	temp_mat.convertTo(temp_mat, CV_8UC1);	

	return temp_mat;
}
