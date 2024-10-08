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

#define EPS 0.001F
#define INF 1E+10F
#define UNKNOWN_FLOW_THRESH 1e9

class MyDis{
    public:
        MyDis();
        ~MyDis();

        Mat run(Mat src1, Mat src2);
        void motionToColor(Mat flow, Mat &color);

    private:
        void prepareBuffers(Mat &I0, Mat &I1, Mat &flow);
        void precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &dst_I0x, Mat &dst_I0y, Mat &I0x, Mat &I0y);
        void PatchInverseSearch_ParBody(int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                    Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1,                                                                                                                          
                                    Mat &_I0x, Mat &_I0y, int _num_iter, int _pyr_level);
        float computeSSDMeanNorm(uchar *I0_ptr, uchar *I1_ptr, int I0_stride, int I1_stride, float w00, float w01,
                                float w10, float w11, int patch_sz);
        float processPatchMeanNorm(float &dst_dUx, float &dst_dUy, uchar *I0_ptr, uchar *I1_ptr, short *I0x_ptr,
                                   short *I0y_ptr, int I0_stride, int I1_stride, float w00, float w01, float w10,
                                   float w11, int patch_sz, float x_grad_sum, float y_grad_sum);
        void Densification_ParBody(int _h, Mat &dst_Ux, Mat &dst_Uy,                                                                                                                                        
                                    Mat &src_Sx, Mat &src_Sy, Mat &_I0, Mat &_I1);

    private:
        int finest_scale;
        int coarsest_scale;
        int patch_size;
        int patch_stride;
        int grad_descent_iter;
        int variational_refinement_iter;
        float variational_refinement_alpha;
        float variational_refinement_gamma;
        float variational_refinement_delta;
        int border_size;
        int w, h;
        int ws, hs;


        vector<Mat> I0s;     //!< Gaussian pyramid for the current frame
        vector<Mat> I1s;     //!< Gaussian pyramid for the next frame
        vector<Mat> I1s_ext; //!< I1s with borders

        vector<Mat> I0xs; //!< Gaussian pyramid for the x gradient of the current frame
        vector<Mat> I0ys; //!< Gaussian pyramid for the y gradient of the current frame

        vector<Mat> Ux; //!< x component of the flow vectors
        vector<Mat> Uy; //!< y component of the flow vectors

        vector<Mat> initial_Ux; //!< x component of the initial flow field, if one was passed as an input
        vector<Mat> initial_Uy; //!< y component of the initial flow field, if one was passed as an input

        Mat U; //!< a buffer for the merged flow
        Mat Sx; //!< intermediate sparse flow representation (x component)
        Mat Sy; //!< intermediate sparse flow representation (y component)

        /* Structure tensor components: */
        Mat I0xx_buf; //!< sum of squares of x gradient values
        Mat I0yy_buf; //!< sum of squares of y gradient values
        Mat I0xy_buf; //!< sum of x and y gradient products

        /* Extra buffers that are useful if patch mean-normalization is used: */
        Mat I0x_buf; //!< sum of x gradient values
        Mat I0y_buf; //!< sum of y gradient values

        /* Auxiliary buffers used in structure tensor computation: */
        Mat I0xx_buf_aux;
        Mat I0yy_buf_aux;
        Mat I0xy_buf_aux;
        Mat I0x_buf_aux;
        Mat I0y_buf_aux;
};
