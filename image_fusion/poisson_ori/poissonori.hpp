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

class MyPoissonOriTest{
    public:
        MyPoissonOriTest();
        ~MyPoissonOriTest();

		Mat Run(Mat img1, Mat img2, Rect ROI, int posX, int posY);
	
	private:
		Mat GetResult(Mat A, Mat B, Rect ROI);
		Mat GetB(Mat img1, Mat img2, int posX, int posY, Rect ROI);
		Mat GetLaplacian();
		Mat GetA(int height, int width);
		int GetLabel(int i, int j, int height, int width);
};
