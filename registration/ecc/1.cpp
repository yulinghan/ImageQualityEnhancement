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
#include "ecc.hpp"

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

    Mat warp_matrix;
    MyEccTest *my_ecc_test = new MyEccTest();
    double rho = my_ecc_test->findTransformECC(ref, input, warp_matrix, MOTION_AFFINE, number_of_iterations, termination_eps, gaussFiltSize);

    cout << "rho:" << rho << endl;
    cout << "warp_matrix:" << endl;
    cout << warp_matrix << endl;

    int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    Mat warped;
    warpAffine(input, warped, warp_matrix, input.size(), imageFlags);
    imwrite("warped.jpg", warped);

	return 0;
}
