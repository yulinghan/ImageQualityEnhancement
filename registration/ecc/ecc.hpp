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

class MyEccTest{
    public:
        MyEccTest();
        ~MyEccTest();

        double findTransformECC(Mat ref, Mat input, Mat &warpMatrix,
                int motionType, int number_of_iterations, float termination_eps, int gaussFiltSize);
    private:
        void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2,
                                    const Mat& src3, const Mat& src4,
                                    const Mat& src5, Mat& dst);

        void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2,
                                         const Mat& src3, const Mat& src4,
                                         const Mat& src5, Mat& dst);

        void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2,
                const Mat& src3, const Mat& src4, Mat& dst);

        void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst);

        void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst);

        void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType);

        double computeECC(InputArray templateImage, InputArray inputImage, InputArray inputMask);

};
