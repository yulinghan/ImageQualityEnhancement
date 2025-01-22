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
#include <omp.h>

using namespace cv;
using namespace std;

class PoissonMatting{
    public:
        PoissonMatting();
        ~PoissonMatting();

		Mat Run(Mat src, Mat alpha_mat);

    private:
        vector<Point> findBoundaryPixels(Mat trimap, int a, int b);
        Mat matting(Mat _image, Mat _trimap);
};
