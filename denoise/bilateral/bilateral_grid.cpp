#include "bilateral_grid.hpp"

MyBilateralGridTest::MyBilateralGridTest() {
}

MyBilateralGridTest::~MyBilateralGridTest() {
}

int *** MyBilateralGridTest::qx_alloci(int n,int r,int c) {
    int *a,**p,***pp;
    int rc=r*c;
    int i,j;

    a = (int*)malloc(sizeof(int)*(n*rc));

    if(a==NULL) {
        printf("qx_allocu_3() fail, Memory is too huge, fail.\n"); 
        getchar(); 
        exit(0); 
    }
    p  = (int**)malloc(sizeof(int*)*n*r);
    pp = (int***)malloc(sizeof(int**)*n);
    for(i=0;i<n;i++) {
        for(j=0;j<r;j++) { 
            p[i*r+j]=&a[i*rc+j*c];
        }
    }
    for(i=0;i<n;i++) {
        pp[i]=&p[i*r];
    }

    return pp;
}                 

vector<vector<vector<float>>> MyBilateralGridTest::Cal3DGaussianTemplate(int r, float gauss_sigma, float value_sigma) {
    int center = r;
    int ksize = r*2+1;
    float x2, y2, z2;
    vector<vector<vector<float>>> Kore;

    for (int i=0; i<ksize;i++) {
        x2 = pow(i - center, 2);
        vector<vector<float>> Kore1;
        for (int j=0; j<ksize; j++) {
            y2 = pow(j - center, 2);
            vector<float> Kore2;
            for (int z=0; z<ksize; z++) {
                z2 = pow(z - center, 2);
                double g = exp(-0.5*((x2+y2)/(2*gauss_sigma*gauss_sigma))*(z2/(2*value_sigma*value_sigma)));
                Kore2.push_back(g);
            }
            Kore1.push_back(Kore2);
        }
        Kore.push_back(Kore1);
    }
    return Kore;
}

Mat MyBilateralGridTest::Run(Mat src, int r, float gauss_sigma, float value_sigma) {
    //设置3d网格窗口大小
    float spaceSample = 16;
    float rangeSample = 8;   
    int grad_x = (int)ceil(src.rows / spaceSample);
    int grad_y = (int)ceil(src.cols / spaceSample);
    int grad_z = (int)ceil(256 / rangeSample);

    //计算3d高斯滤波核
    vector<vector<vector<float>>> gaussian_kore = Cal3DGaussianTemplate(r, gauss_sigma, value_sigma);

    int ***src_grad   = qx_alloci(grad_x, grad_y, grad_z);
    int ***grad_count = qx_alloci(grad_x, grad_y, grad_z);
    int ***dst_grad   = qx_alloci(grad_x, grad_y, grad_z);
    int cur_x, cur_y, cur_z;

    //输入图像拆分到3d网格中
    for(int x=0; x<src.rows; x++) {
        int cur_x = x / spaceSample;
        uchar *ptr_src = src.ptr(x);
        for(int y=0; y<src.cols; y++) {
            int cur_y = y / spaceSample;
            int cur_z = ptr_src[y] / rangeSample;
            src_grad[cur_x][cur_y][cur_z]   += ptr_src[y];
            grad_count[cur_x][cur_y][cur_z] += 1;
        }
    }

    //对3d网格进行高斯滤波
    for(int x=0; x<grad_x; x++) {
        for(int y=0; y<grad_y; y++) {
            for(int z=0; z<grad_z; z++) {
                float value = 0.0, weight = 0.0;
                for(int m1=-r; m1<r; m1++) {
                    int cur_m1 = max(min(x+m1, grad_x-1), 0);
                    for(int m2=-r; m2<r; m2++) {
                        int cur_m2 = max(min(y+m2, grad_y-1), 0);
                        for(int m3=-r; m3<r; m3++) {
                            int cur_m3 = max(min(z+m3, grad_z-1), 0);
                            value += src_grad[cur_m1][cur_m2][cur_m3] * gaussian_kore[r+m1][r+m2][r+m3];
                            weight += gaussian_kore[r+m1][r+m2][r+m3] * grad_count[cur_m1][cur_m2][cur_m3];
                        }
                    }
                }
                dst_grad[x][y][z] = fmin(value / weight, 255.0);
            }
        }
    }

    //从3d网格中恢复滤波后结果图像
    for(int x=0; x<src.rows; x++) {
        int cur_x = x / spaceSample;
        uchar *ptr_src = src.ptr(x);
        for(int y=0; y<src.cols; y++) {
            int cur_y = y / spaceSample;
            int cur_z = ptr_src[y] / rangeSample;
            ptr_src[y] = dst_grad[cur_x][cur_y][cur_z];
        }
    }

	return src;
}
