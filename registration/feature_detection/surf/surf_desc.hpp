#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>  
#include <unistd.h>
#include "common.hpp"                                                                                                                                                                                                           

class MySurfDescTest{
    public:
        MySurfDescTest();
        ~MySurfDescTest();

        void run(Mat src, vector<MyKeyPoint> &key_point_vec);
        Mat DispDesc(Mat src, vector<MyKeyPoint> key_point_vec); 

    private:
        float haarX(Mat img, int row, int column, int s);
        float haarY(Mat img, int row, int column, int s);
        float getAngle(float X, float Y);
        float gaussian(int x, int y, float sig);
        void GetOrientation(Mat integ_mat, MyKeyPoint &ipt);
        void GetDescriptor(Mat integ_mat, MyKeyPoint &ipt);
        int fRound(float flt);

    private:
        float pi = 3.14159f;
        float gauss25[7][7] = 
            {0.02546481,0.02350698,	0.01849125,	0.01239505,	0.00708017,	0.00344629,	0.00142946,
            0.02350698,	0.02169968,	0.01706957,	0.01144208,	0.00653582,	0.00318132,	0.00131956,
            0.01849125,	0.01706957,	0.01342740,	0.00900066,	0.00514126,	0.00250252,	0.00103800,
            0.01239505,	0.01144208,	0.00900066,	0.00603332,	0.00344629,	0.00167749,	0.00069579,
            0.00708017,	0.00653582,	0.00514126,	0.00344629,	0.00196855,	0.00095820,	0.00039744,
            0.00344629,	0.00318132,	0.00250252,	0.00167749,	0.00095820,	0.00046640,	0.00019346,
            0.00142946,	0.00131956,	0.00103800,	0.00069579,	0.00039744,	0.00019346,	0.00008024};
};
