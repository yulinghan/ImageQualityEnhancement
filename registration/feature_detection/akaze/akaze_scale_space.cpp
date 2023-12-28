#include "akaze_scale_space.hpp"

MyAkazeKeyScaleSpaceTest::MyAkazeKeyScaleSpaceTest() {
}

MyAkazeKeyScaleSpaceTest::~MyAkazeKeyScaleSpaceTest() {
}

int MyAkazeKeyScaleSpaceTest::GetGaussianKernelSize(float sigma) {
    int ksize = (int)cvCeil(2.0f*(1.0f + (sigma - 0.8f) / (0.3f)));
    ksize |= 1; // kernel should be odd
    return ksize;
}

float MyAkazeKeyScaleSpaceTest::Compute_K_Percentile(Mat img, int nbins) {
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

Mat MyAkazeKeyScaleSpaceTest::weickert_diffusivity(Mat Lx, Mat Ly, float k) {
    Mat dst = Mat::zeros(Lx.size(), Lx.type());

    float inv_k = 1.0f / (k*k);
    for(int y=0; y<Lx.rows; y++) {
        float* Lx_row = Lx.ptr<float>(y);
        float* Ly_row = Ly.ptr<float>(y);
        float* dst_row = dst.ptr<float>(y);

        for(int x=0; x<Lx.cols; x++) {
            float dL = inv_k*(Lx_row[x]*Lx_row[x] + Ly_row[x]*Ly_row[x]);
            dst_row[x] = -3.315f/(dL*dL*dL*dL);
        }
    }

    exp(dst, dst);
    dst = 1.0 - dst;

    return dst;
}

int MyAkazeKeyScaleSpaceTest::fed_tau_by_process_time(float T, int M, float tau_max, vector<float>& tau) {
    // All cycles have the same fraction of the stopping time
    float t = T/(float)M;
    int n = 0;          // Number of time steps
    float scale = 0.0;  // Ratio of t we search to maximal t

    // Compute necessary number of time steps
    n = cvCeil(sqrtf(3.0f*t/tau_max+0.25f)-0.5f-1.0e-8f);
    scale = 3.0f*t/(tau_max*(float)(n*(n+1)));

    // Call internal FED time step creation routine
    float c = 0.0, d = 0.0;     // Time savers
    vector<float> tauh;    // Helper vector for unsorted taus

    if (n <= 0) {
        return 0;
    }

    // Allocate memory for the time step size
    tau = vector<float>(n);
    tauh = vector<float>(n);

    // Compute time saver
    c = 1.0f / (4.0f * (float)n + 2.0f);
    d = scale * tau_max / 2.0f;

    // Set up originally ordered tau vector
    for (int k = 0; k < n; ++k) {
        float h = cosf((float)CV_PI * (2.0f * (float)k + 1.0f) * c);
        tauh[k] = d / (h * h);
    }

    // Permute list of time steps according to chosen reordering function
    int kappa = 0, prime = 0;
    // Choose kappa cycle with k = n/2
    // This is a heuristic. We can use Leja ordering instead!!
    kappa = n / 2;

    // Get modulus for permutation
    prime = n + 1;

    while (!fed_is_prime_internal(prime)) {
        prime++;
    }

    // Perform permutation
    for(int k = 0, l = 0; l < n; ++k, ++l) {
        int index = 0;
        while ((index = ((k+1)*kappa) % prime - 1) >= n) {
            k++;
        }
        tau[l] = tauh[index];
    }

    return n;
}

bool MyAkazeKeyScaleSpaceTest::fed_is_prime_internal(int& number) {
    bool is_prime = false;

    if (number <= 1) {
        return false;
    } else if (number == 1 || number == 2 || number == 3 || number == 5 || number == 7) {
        return true;
    } else if ((number % 2) == 0 || (number % 3) == 0 || (number % 5) == 0 || (number % 7) == 0) {
        return false;
    } else {
        is_prime = true;
        int upperLimit = (int)sqrt(1.0f + number);
        int divisor = 11;

        while (divisor <= upperLimit ) {
            if (number % divisor == 0) {
                is_prime = false;
            }
            divisor +=2;
        }
        return is_prime;
    }
}

void MyAkazeKeyScaleSpaceTest::nld_step_scalar_one_lane(Mat Lt, Mat Lf, Mat& Lstep, float step_size) {
    /* The labeling scheme for this five star stencil:
       [    a    ]
       [ -1 c +1 ]
       [    b    ]
     */

    Lstep.create(Lt.size(), Lt.type());
    const int cols = Lt.cols - 2;
    int row = 0;

    const float *lt_a, *lt_c, *lt_b;
    const float *lf_a, *lf_c, *lf_b;
    float *dst;
    float step_r = 0.f;

    // Process the top row
    lt_c = Lt.ptr<float>(0) + 1;  /* Skip the left-most column by +1 */
    lf_c = Lf.ptr<float>(0) + 1;
    lt_b = Lt.ptr<float>(1) + 1;
    lf_b = Lf.ptr<float>(1) + 1;

    // fill the corner to prevent uninitialized values
    dst = Lstep.ptr<float>(0);
    dst[0] = 0.0f;
    ++dst;

    for (int j = 0; j < cols; j++) {
        step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
            (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
            (lf_c[j] + lf_b[j    ])*(lt_b[j    ] - lt_c[j]);
        dst[j] = step_r * step_size;
    }

    // fill the corner to prevent uninitialized values
    dst[cols] = 0.0f;
    ++row;

    // Process the middle rows
    int middle_end = Lt.rows - 1;
    for (; row < middle_end; ++row) {
        lt_a = Lt.ptr<float>(row - 1);
        lf_a = Lf.ptr<float>(row - 1);
        lt_c = Lt.ptr<float>(row    );
        lf_c = Lf.ptr<float>(row    );
        lt_b = Lt.ptr<float>(row + 1);
        lf_b = Lf.ptr<float>(row + 1);
        dst = Lstep.ptr<float>(row);

        // The left-most column
        step_r = (lf_c[0] + lf_c[1])*(lt_c[1] - lt_c[0]) +
            (lf_c[0] + lf_b[0])*(lt_b[0] - lt_c[0]) +
            (lf_c[0] + lf_a[0])*(lt_a[0] - lt_c[0]);
        dst[0] = step_r * step_size;

        lt_a++; lt_c++; lt_b++;
        lf_a++; lf_c++; lf_b++;
        dst++;

        // The middle columns
        for (int j = 0; j < cols; j++) {
            step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
                (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
                (lf_c[j] + lf_b[j    ])*(lt_b[j    ] - lt_c[j]) +
                (lf_c[j] + lf_a[j    ])*(lt_a[j    ] - lt_c[j]);
            dst[j] = step_r * step_size;
        }

        // The right-most column
        step_r = (lf_c[cols] + lf_c[cols - 1])*(lt_c[cols - 1] - lt_c[cols]) +
            (lf_c[cols] + lf_b[cols    ])*(lt_b[cols    ] - lt_c[cols]) +
            (lf_c[cols] + lf_a[cols    ])*(lt_a[cols    ] - lt_c[cols]);
        dst[cols] = step_r * step_size;
    }

    // Process the bottom row (row == Lt.rows - 1)
    lt_a = Lt.ptr<float>(row - 1) + 1;  /* Skip the left-most column by +1 */
    lf_a = Lf.ptr<float>(row - 1) + 1;
    lt_c = Lt.ptr<float>(row    ) + 1;
    lf_c = Lf.ptr<float>(row    ) + 1;

    // fill the corner to prevent uninitialized values
    dst = Lstep.ptr<float>(row);
    dst[0] = 0.0f;
    ++dst;

    for (int j = 0; j < cols; j++) {
        step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
            (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
            (lf_c[j] + lf_a[j    ])*(lt_a[j    ] - lt_c[j]);
        dst[j] = step_r * step_size;
    }

    // fill the corner to prevent uninitialized values
    dst[cols] = 0.0f;
}

vector<vector<Mat>> MyAkazeKeyScaleSpaceTest::create_nonlinear_scale_space(Mat img) {
    float soffset = 1.6f;
    int nOctaves = 4;
    int nOctaveLayers = 4;
    vector<vector<Mat>> Lsmooth_arr2;
    vector<vector<Mat>> Lt_arr2;
    vector<vector<float>> etime_arr2;

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // create first level of the evolution
    int ksize = GetGaussianKernelSize(soffset);

    Mat lsmooth, lt;
    GaussianBlur(img, lsmooth, Size(ksize, ksize), soffset, soffset, BORDER_REPLICATE);
    lsmooth.copyTo(lt);

    // compute the kcontrast factor
    GaussianBlur(img, lsmooth, Size(5, 5), 1.0f, 1.0f, BORDER_REPLICATE);
    float kcontrast = Compute_K_Percentile(lsmooth, 128);

    for(int i=0; i<nOctaves; i++) {
        vector<float> etime_arr;
        for(int j=0; j<nOctaveLayers; j++) {
            float esigma = soffset*pow(2.f, (float)(j) / (float)(nOctaveLayers) + i);
            float etime =  0.5f * (esigma * esigma);

            etime_arr.push_back(etime);
        }
        etime_arr2.push_back(etime_arr);
    }

    // Now generate the rest of evolution levels
    float ttime1, ttime2;
    Mat cur_lt, cur_lsmooth;

    for(int i=0; i<nOctaves; i++) {
        vector<Mat> Lsmooth_arr, Lt_arr;
        if(i==0) {
            cur_lt = lt;
            ttime1 = etime_arr2[0][0];
        } else {
            resize(Lt_arr2[i-1][nOctaveLayers-1], cur_lt, Lt_arr2[i-1][nOctaveLayers-1].size()/2);
            kcontrast *= 0.75f;
            ttime1 = etime_arr2[i-1][nOctaveLayers-1];
        }
        for(int j=0; j<nOctaveLayers; j++) {
            Mat cur_lsmooth;
            GaussianBlur(cur_lt, cur_lsmooth, Size(5, 5), 1.0f, 1.0f, BORDER_REPLICATE);
            Lsmooth_arr.push_back(cur_lsmooth);
            
            float k1[]={1, -1}, k2[3][1]={1, -1};
            Mat Kore1 = Mat(1, 2, CV_32FC1,k1);
            Mat Kore2 = Mat(2, 1, CV_32FC1,k2);
            Point point1(-1, 0);
            Point point2(0, -1);

            Mat Lx, Ly;
            filter2D(cur_lsmooth, Lx, -1, Kore1, point1, 0, BORDER_CONSTANT);
            filter2D(cur_lsmooth, Ly, -1, Kore2, point2, 0, BORDER_CONSTANT);

            Mat Lflow = weickert_diffusivity(Lx, Ly, kcontrast);

            vector<float> tau;
            ttime2 = etime_arr2[i][j];
            float ttime = ttime2 - ttime1;
            ttime1 = ttime2;
            int naux = fed_tau_by_process_time(ttime, 1, 0.25f, tau);

            // Perform Fast Explicit Diffusion on Lt
            for (int m=0; m<tau.size(); m++) {
                float step_size = tau[m] * 0.5f;
                Mat Lstep;
    
                nld_step_scalar_one_lane(cur_lt, Lflow, Lstep, step_size);
                add(cur_lt, Lstep, cur_lt);
            }
            Lt_arr.push_back(cur_lt.clone());
            Lsmooth_arr.push_back(cur_lsmooth.clone());
        }
        Lt_arr2.push_back(Lt_arr);
        Lsmooth_arr2.push_back(Lsmooth_arr);
    }

    return Lt_arr2;
}

vector<vector<Mat>> MyAkazeKeyScaleSpaceTest::run(Mat src) {
    vector<vector<Mat>> Lt_arr2 = create_nonlinear_scale_space(src);

    return Lt_arr2;
}
