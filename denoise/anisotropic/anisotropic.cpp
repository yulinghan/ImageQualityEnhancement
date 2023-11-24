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



Mat MyAnisotropicTest::Run(Mat src) {
    Mat anisotropic_mat;
    Mat temp_mat = src.clone();

    int iterations=30;
    float lambda=0.15;
    float k=10;

    for(int t = 0;t < iterations; ++t) {
        anisotropic_mat = temp_mat.clone();
        for(int i=1; i<src.rows-1; i++) {
            uchar *data_ori_ptr  = anisotropic_mat.ptr(i);
            uchar *data_prev_ptr = anisotropic_mat.ptr(i-1);
            uchar *data_next_ptr = anisotropic_mat.ptr(i+1);
            uchar *temp_mat_ptr  = temp_mat.ptr(i);

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
                temp_mat_ptr[j] = fmin(fmax(value, 0.0), 255.0);
            }
        }
    }

	return temp_mat;
}
