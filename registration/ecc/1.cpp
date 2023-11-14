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
#include "fa_ecc.hpp"
#include "ic_ecc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    Mat input = imread(argv[1], 0);
    Mat ref   = imread(argv[2], 0);

    resize(input, input, input.size()/4);
    resize(ref, ref, input.size());

    imwrite("input.jpg", input);
    imwrite("ref.jpg", ref);

    int number_of_iterations = 100;
    double termination_eps = 1e-8;
    int gaussFiltSize = 3;
    int motionType = MOTION_TRANSLATION;
    int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;

    //前向累加fa_ecc
    Mat warp_matrix;
    MyFaEccTest *my_fa_ecc_test = new MyFaEccTest();
    double rho = my_fa_ecc_test->findTransformECC(ref, input, warp_matrix, motionType, number_of_iterations, termination_eps, gaussFiltSize);

    cout << "fa_rho:" << rho << endl;
    cout << "fa_warp_matrix:" << endl;
    cout << warp_matrix << endl;

    Mat warped;
    warpAffine(input, warped, warp_matrix, input.size(), imageFlags);
    imwrite("fa_warped.jpg", warped);

    //反向累加ic_ecc
    MyIcEccTest *my_ic_ecc_test = new MyIcEccTest();
    rho = my_ic_ecc_test->findTransformECC(ref, input, warp_matrix, motionType, number_of_iterations, termination_eps, gaussFiltSize);

    cout << "ic_rho:" << rho << endl;
    cout << "ic_warp_matrix:" << endl;
    cout << warp_matrix << endl;

    warpAffine(input, warped, warp_matrix, input.size(), imageFlags);
    imwrite("ic_warped.jpg", warped);



	return 0;
}
