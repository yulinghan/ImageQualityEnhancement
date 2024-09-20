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

class BanterleTMO{
    public:
        BanterleTMO();
        ~BanterleTMO();

		Mat Run(Mat src);

    private:
        vector<int> GetHdrToMergeNum(Mat seg_mat, Mat imgBin, int seg_num);
        Mat GetSegMerge(Mat seg_mat, int seg_num, float thres, vector<int> seg_pixel_num, vector<vector<int>> seg_neighbor_arr2);
        vector<vector<int>> GetSegNeighbor(Mat seg_mat, int seg_num);
        Mat GetHdrSeg(Mat src_blur);
        vector<int> GetSegAreaNum(Mat seg_mat, int seg_num);
        Mat seg_area(Mat imgBin_ori, int &seg_num);
        void region_grow(Mat imgBin, int x, int y, int cur_value, vector<int> &addr_array_x, vector<int> &addr_array_y);
        float sign(float x);
};
