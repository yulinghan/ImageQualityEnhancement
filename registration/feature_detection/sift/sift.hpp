#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>  
#include <unistd.h>
                                                                                                                                                                                                           
using namespace cv; 
using namespace std;

struct KeypointSt {  
    float row, col; // 反馈回原图像大小，特征点的位置
    float sx,sy;    // 金字塔中特征点的位置
    int octave,level;//金字塔中，特征点所在的阶梯、层次

    float scale,ori,mag; //所在层的尺度sigma,主方向orientation (range [-PI,PI])，以及幅值
    float descrip[128];       //特征描述字指针：128维或32维等
};

class MySiftTest{
    public:
        MySiftTest();
        ~MySiftTest();

        void run(Mat input);

    private:
        void BuildGaussianDogPyr(Mat gauss_mat, int numoctaves,
                    vector<vector<Mat>> &gaussian_pyr,
                    vector<vector<Mat>> &dog_pyr,
                    vector<vector<int>> &kern_size_arr);
        Mat ScaleInitImage(Mat src, int filter_size);

        int DetectKeypoint(int numoctaves,
                    vector<vector<int>> kern_size_arr,
                    vector<vector<Mat>> gaussian_pyr,
                    vector<vector<Mat>> dog_pyr,
                    vector<KeypointSt> &KeypointSt_vec);

    private:
        int SCALESPEROCTAVE = 2;
        float CURVATURE_THRESHOLD = 10.0;
        float CONTRAST_THRESHOLD = 0.02;
};
