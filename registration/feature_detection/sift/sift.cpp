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

int MySiftTest::DetectKeypoint(int numoctaves, 
                    vector<vector<int>> kern_size_arr,
                    vector<vector<Mat>> gaussian_pyr,
                    vector<vector<Mat>> dog_pyr,
                    vector<KeypointSt> &KeypointSt_vec) {

    //计算用于DOG极值点检测的主曲率比的阈值  
	double curvature_threshold = curvature_threshold = ((CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1))/CURVATURE_THRESHOLD;  
	int keypoint_count = 0;

    for (int i=0; i<numoctaves; i++) {          
        for(int j=1;j<SCALESPEROCTAVE+1;j++)//取中间的scaleperoctave个层  
        {    
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

void MySiftTest::run(Mat src) {
    src.convertTo(src, CV_32FC1, 1/255.0);

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

    vector<KeypointSt> KeypointSt_vec;
    int keypoint_count = DetectKeypoint(numoctaves, kern_size_arr, gauss_pyr, dog_pyr, KeypointSt_vec);
    cout << "keypoint_count:" << keypoint_count << endl;

#if 0
    for(int i=0; i<keypoint_count; i++) {
        Point p(KeypointSt_vec[i].col, KeypointSt_vec[i].row);//初始化点坐标为(20,20)
        circle(src, p, 3, Scalar(255, 255, 255), -1);  // 画半径为1的圆(画点）
    }

    imshow("src", src);
    waitKey(0);
#endif

}
