#include "sift.hpp"

MySiftTest::MySiftTest() {
}

MySiftTest::~MySiftTest() {
}

Mat MySiftTest::ScaleInitImage(Mat src, int filter_size) {
    Mat dst;
    GaussianBlur(src, dst, Size(filter_size, filter_size), 0, 0);

    return dst;
}

void MySiftTest::BuildGaussianDogPyr(Mat gauss_mat, int numoctaves,
                    vector<vector<Mat>> &gaussian_pyr, 
                    vector<vector<Mat>> &dog_pyr,
                    vector<vector<int>> &kern_size_arr) {
    double k = pow(2, 1.0/((float)SCALESPEROCTAVE));
    int init_kern_size = 3*sqrt(k*k-1);

    Mat cur_src = gauss_mat;
    for(int i=0; i<numoctaves; i++) {
        int cur_kern_size = init_kern_size;
        vector<Mat> gauss_vec, dog_vec;
        gauss_vec.push_back(cur_src);
        vector<int> kern_size_vec;

        for (int j=1; j<SCALESPEROCTAVE+3; j++) {
            Mat cur_gauss, cur_dog;
            GaussianBlur(gauss_vec[j-1], cur_gauss, Size(cur_kern_size*2+1, cur_kern_size*2+1), 0, 0);
            gauss_vec.push_back(cur_gauss);
            
            cur_dog = gauss_vec[j-1] - gauss_vec[j];
            dog_vec.push_back(cur_dog);
            kern_size_vec.push_back(cur_kern_size);

            cur_kern_size = k*cur_kern_size;
        }
        resize(cur_src, cur_src, cur_src.size()/2);

        gaussian_pyr.push_back(gauss_vec);
        dog_pyr.push_back(dog_vec);
        kern_size_arr.push_back(kern_size_vec);
    }
}

int MySiftTest::DetectKeypoint(vector<vector<int>> kern_size_arr,
                            vector<vector<Mat>> dog_pyr,
                            vector<KeypointSt> &KeypointSt_vec) {

    //计算用于DOG极值点检测的主曲率比的阈值  
	double curvature_threshold = curvature_threshold = ((CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1))/CURVATURE_THRESHOLD;  
	int keypoint_count = 0;

    for (int i=0; i<dog_pyr.size(); i++) {          
        for(int j=1;j<SCALESPEROCTAVE+1;j++) { //取中间的scaleperoctave个层  
            //在图像的有效区域内寻找具有显著性特征的局部最大值  
            int dim = kern_size_arr[i][j];
            for (int m=dim;m<(dog_pyr[i][0].rows-dim);m++)  {
                for(int n=dim;n<(dog_pyr[i][0].cols-dim);n++) {
                    float inf_val = dog_pyr[i][j].at<float>(m, n);  
                    if (fabs(inf_val)>= CONTRAST_THRESHOLD ) {

                        int total_num = 0;
                        for(int k1=-1; k1<=1; k1++) {
                            for(int k2=-2; k2<=1; k2++) {
                                for(int k3=-1; k3<=1; k3++) {
                                    if(inf_val <= dog_pyr[i][j+k1].at<float>(m+k2, n+k3)) {
                                        total_num = total_num - 1;
                                    }
                                    if(inf_val >= dog_pyr[i][j+k1].at<float>(m+k2, n+k3)) {
                                        total_num = total_num + 1;
                                    }
                                }
                            }
                        }
                        if(abs(total_num) == 27) {
                            float Dxx,Dyy,Dxy,Tr_H,Det_H,curvature_ratio;  
                            Dxx = dog_pyr[i][j].at<float>(m, n-1) + dog_pyr[i][j].at<float>(m, n+1) - 2*dog_pyr[i][j].at<float>(m, n);
                            Dyy = dog_pyr[i][j].at<float>(m-1, n) + dog_pyr[i][j].at<float>(m+1, n) - 2*dog_pyr[i][j].at<float>(m, n);
                            Dxy = dog_pyr[i][j].at<float>(m-1, n-1) + dog_pyr[i][j].at<float>(m+1, n+1) 
                                - dog_pyr[i][j].at<float>(m+1, n-1) - dog_pyr[i][j].at<float>(m-1, n+1);

                            Tr_H = Dxx + Dyy;  
                            Det_H = Dxx*Dyy - Dxy*Dxy;  
                            curvature_ratio = (1.0*Tr_H*Tr_H)/Det_H;  

                            //最后得到最具有显著性特征的特征点
                            if ((Det_H>=0.0) && (curvature_ratio <= curvature_threshold) ) {
                                keypoint_count++;  

                                KeypointSt k;
                                k.row = m*pow(2, i);
                                k.col = n*pow(2, i);
                                k.sy = m;
                                k.sx = n;
                                k.octave=i;  
                                k.level=j;  
                                KeypointSt_vec.push_back(k);
                            }
                        }
                    }
                }
            }
        }
    }
    return keypoint_count;  
}

int MySiftTest::FindClosestRotationBin(int binCount, float angle) {  
	angle += CV_PI;  
	angle /= 2.0 * CV_PI;  
	// calculate the aligned bin  
	angle *= binCount;  
	int idx = (int) angle;  
	if (idx == binCount)  
		idx = 0;  
	return (idx);  
}  

void MySiftTest::AverageWeakBins(double* hist, int binCount) {
	for (int sn = 0 ; sn < 2 ; ++sn) {
		double firstE = hist[0];
		double last = hist[binCount-1];
		for (int sw = 0 ; sw < binCount ; ++sw) {
			double cur = hist[sw];
			double next = (sw == (binCount - 1)) ? firstE : hist[(sw + 1) % binCount];
			hist[sw] = (last + cur + next) / 3.0;
			last = cur;
		}
	}
}

bool MySiftTest::InterpolateOrientation(double left, double middle,double right, double *degreeCorrection, double *peakValue)  {
	double a = ((left + right) - 2.0 * middle) / 2.0;   //抛物线捏合系数a  

	if (a == 0.0)  
		return false;  
	double c = (((left - middle) / a) - 1.0) / 2.0;  
	double b = middle - c * c * a;  
	if (c < -0.5 || c > 0.5)  
		return false;  
	*degreeCorrection = c;  
	*peakValue = b;  
	return true;  
}  

int MySiftTest::AssignTheMainOrientation(vector<vector<Mat>> gauss_pyr, vector<vector<int>> kern_size_arr, 
                                vector<KeypointSt> KeypointSt_vec_1, vector<KeypointSt> &KeypointSt_vec_2) {
	int num_bins = 36;  
	float hist_step = 2.0*PI/num_bins;  
	float hist_orient[36];  
	for (int i=0;i<36;i++) {
		hist_orient[i]=-PI+i*hist_step;
    }

	int zero_pad = kern_size_arr[0][SCALESPEROCTAVE];
	int keypoint_count = 0;  

	for(int k=0; k<KeypointSt_vec_1.size(); k++){  
		int i=KeypointSt_vec_1[k].octave;
		int j=KeypointSt_vec_1[k].level;  
		int m=KeypointSt_vec_1[k].sy;
		int n=KeypointSt_vec_1[k].sx;
		if ((m>=zero_pad)&&(m<gauss_pyr[i][0].rows-zero_pad)&&  
			    (n>=zero_pad)&&(n<gauss_pyr[i][0].cols-zero_pad)) {

			//产生二维高斯模板  
            int cur_kern_size = kern_size_arr[i][j];
            Mat kernelX = getGaussianKernel(cur_kern_size, 1);  
            Mat kernelY = getGaussianKernel(cur_kern_size, 1);
            Mat mat = kernelX * kernelY.t();
			int dim=(int)(0.5 * (mat.rows));

			//分配用于存储Patch幅值和方向的空间  
			//声明方向直方图变量  
			double orienthist[36];
			for(int sw=0; sw<36; ++sw) {
				orienthist[sw]=0.0;    
			}

			//将梯度方向为了36个，在特征点的周围根据梯度方向,累加梯度幅值
			for (int x=m-dim,mm=0;x<=(m+dim);x++,mm++) {  
				for(int y=n-dim,nn=0;y<=(n+dim);y++,nn++) {
					double dx = 0.5*(gauss_pyr[i][j].at<float>(x, y+1) - gauss_pyr[i][j].at<float>(x, y-1));
					double dy = 0.5*(gauss_pyr[i][j].at<float>(x+1, y) - gauss_pyr[i][j].at<float>(x-1, y));

					//计算特征点处的幅值  
					double mag = sqrt(dx*dx+dy*dy);
					//计算方向  
					double Ori =atan( 1.0*dy/dx);  
					int binIdx = FindClosestRotationBin(36, Ori);                   //得到离现有方向最近的直方块  
					orienthist[binIdx] = orienthist[binIdx] + 1.0* mag * mat.at<float>(mm,nn);//利用高斯加权累加进直方图相应的块  
				}  
            }
            //对36个不同方向梯度幅值，做1维均值滤波平滑
            AverageWeakBins(orienthist, 36);

            //找到最大梯度幅值和它对应梯度方向
            double maxGrad = 0.0;  
            int maxBin = 0;  
            for (int b = 0 ; b < 36 ; ++b) {
                if (orienthist[b] > maxGrad) {
                    maxGrad = orienthist[b];  
                    maxBin = b;  
                }  
            }

            //根据最大梯度幅值，和它前后2两个点，拟合出抛物线曲线，找到这条抛物线上实际最大的副值
            double maxPeakValue=0.0;  
            double maxDegreeCorrection=0.0;  
            if ( (InterpolateOrientation ( orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],  
                            orienthist[maxBin], orienthist[(maxBin + 1) % 36],  
                            &maxDegreeCorrection, &maxPeakValue)) == false) {
                printf("BUG: Parabola fitting broken");
            }

            //36个方向里面，除了主方向价加入特征描述之外。
            //该特征位置36个梯度方向上，幅值比0.8*最大幅值大的梯度方向，作为新的特征描述加入特征队列。
            //新加入的特征和原特征相比：位置相同，尺度相同但是方向不同。
            bool binIsKeypoint[36];  
            for(int b = 0 ; b < 36 ; ++b) {  
                binIsKeypoint[b] = false;  
                if (b == maxBin) {  
                    binIsKeypoint[b] = true;  
                    continue;  
                }  
                // Local peaks are, too, in case they fulfill the threshhold  
                if (orienthist[b] < (peakRelThresh * maxPeakValue))  
                    continue;  
                int leftI = (b == 0) ? (36 - 1) : (b - 1);  
                int rightI = (b + 1) % 36;  
                if (orienthist[b] <= orienthist[leftI] || orienthist[b] <= orienthist[rightI])  
                    continue; // no local peak  
                binIsKeypoint[b] = true;  
            }

            // find other possible locations  
            double oneBinRad = (2.0 * PI) / 36;  
            for (int b = 0 ; b < 36 ; ++b) {  
                if (binIsKeypoint[b] == false)  
                    continue;  
                int bLeft = (b == 0) ? (36 - 1) : (b - 1);  
                int bRight = (b + 1) % 36;  
                // Get an interpolated peak direction and value guess.  
                double peakValue;  
                double degreeCorrection;  

                if(InterpolateOrientation(orienthist[bLeft], orienthist[b], orienthist[bRight],  
                                          &degreeCorrection, &peakValue) == false) {
                    printf("BUG: Parabola fitting broken");  
                }  

                double degree = (b + degreeCorrection) * oneBinRad - PI;  
                if (degree < -PI)  
                    degree += 2.0 * PI;  
                else if (degree > PI)  
                    degree -= 2.0 * PI;

                keypoint_count++;

                KeypointSt key_p;
                key_p.row = KeypointSt_vec_1[k].row;
                key_p.col = KeypointSt_vec_1[k].col;
                key_p.sy  = KeypointSt_vec_1[k].sy;
                key_p.sx  = KeypointSt_vec_1[k].sx;
                key_p.octave = KeypointSt_vec_1[k].octave;
                key_p.level  = KeypointSt_vec_1[k].level;
                key_p.ori = degree;  
                key_p.mag = peakValue;
                KeypointSt_vec_2.push_back(key_p);
            }
        }
    }
    return keypoint_count;
}

Mat MySiftTest::DisplayKeypointLocation(Mat src, vector<KeypointSt> KeypointSt_vec_1) {
    Mat dst = src.clone();

    for(int i=0; i<KeypointSt_vec_1.size(); i++) {
        Point p(KeypointSt_vec_1[i].col, KeypointSt_vec_1[i].row);//初始化点坐标为(20,20)
        circle(dst, p, 3, Scalar(255, 255, 255), -1);  // 画半径为1的圆(画点）
    }
    return dst;
}

Mat MySiftTest::DisplayOrientation(Mat src, vector<KeypointSt> KeypointSt_vec, vector<vector<int>> kern_size_arr) {
    Mat dst = src.clone();

    for(int i=0; i<KeypointSt_vec.size(); i++) {
		float scale = kern_size_arr[KeypointSt_vec[i].octave][KeypointSt_vec[i].level];
		float autoscale = 3.0;   
		float uu = autoscale*scale*cos(KeypointSt_vec[i].ori);
		float vv = autoscale*scale*sin(KeypointSt_vec[i].ori);
		float x  = (KeypointSt_vec[i].col) + uu;
		float y  = (KeypointSt_vec[i].row) + vv;
		line(dst, Point((int)(KeypointSt_vec[i].col),(int)(KeypointSt_vec[i].row)),   
			Point((int)x,(int)y), CV_RGB(255,255,255), 1, 8, 0 );

		// Arrow head parameters  
		float alpha = 0.33; // Size of arrow head relative to the length of the vector  
		float beta = 0.33;  // Width of the base of the arrow head relative to the length  

		float xx0= (KeypointSt_vec[i].col)+uu-alpha*(uu+beta*vv);  
		float yy0= (KeypointSt_vec[i].row)+vv-alpha*(vv-beta*uu);  
		float xx1= (KeypointSt_vec[i].col)+uu-alpha*(uu-beta*vv);  
		float yy1= (KeypointSt_vec[i].row)+vv-alpha*(vv+beta*uu);  
		line(dst, Point((int)xx0,(int)yy0),   
			Point((int)x,(int)y), CV_RGB(255,255,255), 1, 8, 0 );  
		line(dst, Point((int)xx1,(int)yy1),   
			Point((int)x,(int)y), CV_RGB(255,255,255), 1, 8, 0 );  
	}

    return dst;   
}

float MySiftTest::GetVecNorm( float* vec, int dim) {  
	float sum=0.0;  
	for (unsigned int i=0;i<dim;i++)  
		sum+=vec[i]*vec[i];  
	return sqrt(sum);  
}

void MySiftTest::ExtractFeatureDescriptors(vector<vector<Mat>> gauss_pyr, vector<KeypointSt> KeypointSt_vec) {
	// The orientation histograms have 8 bins  
	float orient_bin_spacing = PI/4;  
	float orient_angles[8]={-PI, -PI+orient_bin_spacing,-PI*0.5, -orient_bin_spacing,  
		                    0.0, orient_bin_spacing, PI*0.5,  PI+orient_bin_spacing};

	//产生描述字中心各点坐标  
	float feat_grid[32];
	for (int i=0; i<GridSpacing; i++) {
		for (int j=0; j<2*GridSpacing; ++j, ++j) {
			feat_grid[i*2*GridSpacing+j]  =-6.0+i*GridSpacing;  
			feat_grid[i*2*GridSpacing+j+1]=-6.0+0.5*j*GridSpacing;  
		}  
	}

	//产生网格  
	float feat_samples[512];
	for (int i=0;i<4*GridSpacing;i++) {
		for (int j=0;j<8*GridSpacing;j+=2)  {
			feat_samples[i*8*GridSpacing+j]  = -(2*GridSpacing-0.5)+i;  
			feat_samples[i*8*GridSpacing+j+1]= -(2*GridSpacing-0.5)+0.5*j;  
		}  
    }

	float feat_window = 2*GridSpacing;  
	for(int k=0; k<KeypointSt_vec.size(); k++){
		float sine = sin(KeypointSt_vec[k].ori);
		float cosine = cos(KeypointSt_vec[k].ori);

		//计算中心点坐标旋转之后的位置  
		float featcenter[32];
		for (int i=0;i<GridSpacing;i++) {
			for (int j=0;j<2*GridSpacing;j+=2) {
				float x=feat_grid[i*2*GridSpacing+j];  
				float y=feat_grid[i*2*GridSpacing+j+1];  
				featcenter[i*2*GridSpacing+j]  = ((cosine*x + sine*y)  + KeypointSt_vec[k].sx);  
				featcenter[i*2*GridSpacing+j+1]= ((-sine*x + cosine*y) + KeypointSt_vec[k].sy);  
			}  
		}  
		// calculate sample window coordinates (rotated along keypoint)  
		float feat[512];
		for(int i=0; i<64*GridSpacing*2; i++,i++) {
			float x = feat_samples[i];  
			float y = feat_samples[i+1];  
			feat[i]=((cosine * x + sine * y) + KeypointSt_vec[k].sx);
			feat[i+1]=((-sine * x + cosine * y) + KeypointSt_vec[k].sy);
		}

		//Initialize the feature descriptor.  
		float feat_desc[128];
		for(int i=0; i<128; i++) {
			feat_desc[i]=0.0;  
		}

		for(int i=0; i<512; ++i,++i) {
			float x_sample = feat[i];  
			float y_sample = feat[i+1];  

			float sample12 = gauss_pyr[KeypointSt_vec[k].octave][KeypointSt_vec[k].level].at<float>(x_sample, y_sample-1);
			float sample21 = gauss_pyr[KeypointSt_vec[k].octave][KeypointSt_vec[k].level].at<float>(x_sample-1, y_sample);
			float sample22 = gauss_pyr[KeypointSt_vec[k].octave][KeypointSt_vec[k].level].at<float>(x_sample, y_sample);
			float sample23 = gauss_pyr[KeypointSt_vec[k].octave][KeypointSt_vec[k].level].at<float>(x_sample+1, y_sample);
			float sample32 = gauss_pyr[KeypointSt_vec[k].octave][KeypointSt_vec[k].level].at<float>(x_sample, y_sample+1);

			float diff_x = sample23 - sample21;  
			float diff_y = sample32 - sample12;  
			float mag_sample  = sqrt(diff_x*diff_x + diff_y*diff_y);
			float grad_sample = atan(diff_y / diff_x);
			if(grad_sample == CV_PI)
				grad_sample = -CV_PI;

			// Compute the weighting for the x and y dimensions.  
			float x_wght[GridSpacing*GridSpacing];
			float y_wght[GridSpacing*GridSpacing]; 
			float pos_wght[8*GridSpacing*GridSpacing];
			for (int m=0;m<32;++m,++m) {
				float x=featcenter[m];  
				float y=featcenter[m+1];  
				x_wght[m/2] = fmax(1 - (fabs(x - x_sample)*1.0/GridSpacing), 0.0);
				y_wght[m/2] = fmax(1 - (fabs(y - y_sample)*1.0/GridSpacing), 0.0);
			}  
			for(int m=0;m<16;++m) {
				for (int n=0;n<8;++n) {
					pos_wght[m*8+n]=x_wght[m]*y_wght[m];
                }
            }

			//计算方向的加权，首先旋转梯度场到主方向，然后计算差异   
			float diff[8],orient_wght[128];  
			for(int m=0;m<8;++m) {
				float angle = grad_sample - (KeypointSt_vec[k].ori) - orient_angles[m]+CV_PI;  
				float temp = angle / (2.0 * CV_PI);  
				angle -= (int)(temp) * (2.0 * CV_PI);  
				diff[m]= angle - CV_PI;  
			}

			// Compute the gaussian weighting.  
			float x = KeypointSt_vec[k].sx;  
			float y = KeypointSt_vec[k].sy;  
			float g = exp(-((x_sample-x)*(x_sample-x)+(y_sample-y)*(y_sample-y))/(2*feat_window*feat_window))/(2*CV_PI*feat_window*feat_window);  

			for(int m=0;m<128;++m) {
				orient_wght[m] = fmax((1.0 - 1.0*fabs(diff[m%8])/orient_bin_spacing), 0.0);
				feat_desc[m] = feat_desc[m] + orient_wght[m]*pos_wght[m]*g*mag_sample;  
			}  
		}  
		float norm = GetVecNorm(feat_desc, 128);  
		for(int m=0;m<128;m++) {
			feat_desc[m]/=norm;  
			if (feat_desc[m]>0.2)  
				feat_desc[m]=0.2;  
		}  
		norm=GetVecNorm( feat_desc, 128);  
		for(int m=0;m<128;m++) {
			feat_desc[m]/=norm;  
            KeypointSt_vec[k].descrip[m] = feat_desc[m] / norm;
		}
	}  
}

void MySiftTest::run(Mat src) {
    src.convertTo(src, CV_32FC1);

    //预滤波除噪声
    int filter_size = 5;
    Mat gauss_mat = ScaleInitImage(src, filter_size);

    //建立Guassian金字塔和DOG金字塔
    int dim = min(gauss_mat.rows, gauss_mat.cols);
    int numoctaves = (int)(log((double)dim)/log(2.0)) - 2;

    vector<vector<Mat>> gauss_pyr, dog_pyr;
    vector<vector<int>> kern_size_arr;
    BuildGaussianDogPyr(gauss_mat, numoctaves, gauss_pyr, dog_pyr, kern_size_arr);

#if 0
    for(int i=0; i<gauss_pyr.size(); i++) {
        for(int j=0; j<gauss_pyr[i].size()-1; j++) {
            imshow(format("%d_%d_gauss", i, j), gauss_pyr[i][j]);
            imshow(format("%d_%d_dog", i, j), dog_pyr[i][j]);
            waitKey(0);
        }
    }
#endif

    //dog图像上进行特征点位置检测
    vector<KeypointSt> KeypointSt_vec_1;
    int keypoint_count_1 = DetectKeypoint(kern_size_arr, dog_pyr, KeypointSt_vec_1);
    cout << "keypoint_count_1:" << keypoint_count_1 << endl;

#if 1
    //显示极值点位置
    Mat Keypoint_show = DisplayKeypointLocation(src, KeypointSt_vec_1);
    imshow("Keypoint", Keypoint_show/255);
#endif

    //计算各个特征点的主方向
    //将各特征点上，主方向子之外，幅值>0.8*主方向幅值的，作为新特征点
    //新特征点这之前特征相比：位置相同，尺度相同但是方向不同
    vector<KeypointSt> KeypointSt_vec_2;
    int keypoint_count_2 = AssignTheMainOrientation(gauss_pyr, kern_size_arr, KeypointSt_vec_1, KeypointSt_vec_2);
    cout << "keypoint_count_2:" << keypoint_count_2 << endl;

#if 1
    //显示特征点方向和幅值
    Mat mag_grd_show = DisplayOrientation(src, KeypointSt_vec_2, kern_size_arr);
    imshow("mag_grd_show", mag_grd_show/255);
#endif

    //128位特征描述符存在了KeypointSt_vec_2[k]..descrip中
    ExtractFeatureDescriptors(gauss_pyr, KeypointSt_vec_2);
}
