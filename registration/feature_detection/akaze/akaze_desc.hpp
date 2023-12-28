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

class MyAkazeDescTest{
    public:
        MyAkazeDescTest();
        ~MyAkazeDescTest();

        Mat run(vector<vector<Mat>> scale_space_arr, vector<KeyPoint> key_points);

    private:
        void GetScaleSpaceIxIy(vector<vector<Mat>> scale_space_y_arr2,
                                vector<vector<Mat>> &scale_space_Ix_arr2,
                                vector<vector<Mat>> &scale_space_Iy_arr2);

        Mat Get_Upright_MLDB_Full_Descriptor(vector<vector<Mat>> scale_space_y_arr2,
                                        vector<vector<Mat>> &scale_space_Ix_arr2,
                                        vector<vector<Mat>> &scale_space_Iy_arr2,
                                        vector<KeyPoint> key_points);

    private:
        int descriptor_size_ = 64;
        int descriptor_type_ = CV_8UC1;
        int descriptor_pattern_size_ = 10;
};
