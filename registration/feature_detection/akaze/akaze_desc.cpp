#include "akaze_desc.hpp"

MyAkazeDescTest::MyAkazeDescTest() {
}

MyAkazeDescTest::~MyAkazeDescTest() {
}

void MyAkazeDescTest::GetScaleSpaceIxIy(vector<vector<Mat>> scale_space_y_arr2, 
                                        vector<vector<Mat>> &scale_space_Ix_arr2, 
                                        vector<vector<Mat>> &scale_space_Iy_arr2) {
    float k1[]={1, -1}, k2[3][1]={1, -1};
    Mat Kore1 = Mat(1, 2, CV_32FC1,k1);
    Mat Kore2 = Mat(2, 1, CV_32FC1,k2);
    Point point1(-1, 0);
    Point point2(0, -1);

    for(int i=0; i<scale_space_y_arr2.size(); i++) {
        vector<Mat> scale_space_Ix_arr, scale_space_Iy_arr;
        for(int j=0; j<scale_space_y_arr2[i].size(); j++) {
            Mat Ix, Iy;
            filter2D(scale_space_y_arr2[i][j], Ix, -1, Kore1, point1, 0, BORDER_CONSTANT);
            filter2D(scale_space_y_arr2[i][j], Iy, -1, Kore2, point2, 0, BORDER_CONSTANT);

            scale_space_Ix_arr.push_back(Ix);
            scale_space_Iy_arr.push_back(Iy);
        }
        scale_space_Ix_arr2.push_back(scale_space_Ix_arr);
        scale_space_Iy_arr2.push_back(scale_space_Iy_arr);
    }
}

Mat MyAkazeDescTest::Get_Upright_MLDB_Full_Descriptor(vector<vector<Mat>> scale_space_y_arr2, 
                                        vector<vector<Mat>> &scale_space_Ix_arr2, 
                                        vector<vector<Mat>> &scale_space_Iy_arr2,
                                        vector<KeyPoint> key_points) {
    Mat desc = Mat::zeros((int)key_points.size(), descriptor_size_, descriptor_type_);

    // For 2x2 grid, 3x3 grid and 4x4 grid
    int pattern_size = descriptor_pattern_size_;
    int sample_step[3] = {
        pattern_size,
        divUp(pattern_size * 2, 3),
        divUp(pattern_size, 2)
    };

    for(int k=0; k<key_points.size(); k++) {
        uchar *desc_ptr = desc.ptr<uchar>(k);
        KeyPoint kpt = key_points[k];
        //Buffer for the M-LDB descriptor
        int max_channels = 3;
        float values[16*max_channels];

        // Get the information from the keypoint
        float ratio = (float)(1 << kpt.octave);
        kpt.size = 64;
        int scale = cvRound(0.5f*kpt.size / ratio);
        int level = kpt.class_id;
        Mat Lx = scale_space_Ix_arr2[kpt.octave][kpt.class_id];
        Mat Ly = scale_space_Iy_arr2[kpt.octave][kpt.class_id];
        Mat Lt = scale_space_y_arr2[kpt.octave][kpt.class_id];
        float yf = kpt.pt.y;
        float xf = kpt.pt.x;

        // For the three grids
        int dcount1 = 0;
        for (int z = 0; z < 3; z++) {
            int dcount2 = 0;
            int step = sample_step[z];
            for (int i = -pattern_size; i < pattern_size; i += step) {
                for (int j = -pattern_size; j < pattern_size; j += step) {
                    float di = 0.0, dx = 0.0, dy = 0.0;

                    int nsamples = 0;
                    for (int k = 0; k < step; k++) {
                        for (int l = 0; l < step; l++) {

                            // Get the coordinates of the sample point
                            const float sample_y = yf + (l+j)*scale;
                            const float sample_x = xf + (k+i)*scale;

                            const int y1 = cvRound(sample_y);
                            const int x1 = cvRound(sample_x);

                            if (y1 < 0 || y1 >= Lt.rows || x1 < 0 || x1 >= Lt.cols)
                                continue; // Boundaries

                            float ri = Lt.at<float>(y1, x1);
                            float rx = Lx.at<float>(y1, x1);
                            float ry = Ly.at<float>(y1, x1);

                            di += ri;
                            dx += rx;
                            dy += ry;
                            nsamples++;
                        }
                    }

                    if (nsamples > 0) {
                        di = di / nsamples;
                        dx = dx / nsamples;
                        dy = dy / nsamples;
                    }


                    values[dcount2*max_channels + 0] = di;
                    values[dcount2*max_channels + 1] = dx;
                    values[dcount2*max_channels + 2] = dy;
                    dcount2++;
                }
            }
            // Do binary comparison
            int num = (z + 2) * (z + 2);
            for (int i = 0; i < num; i++) {
                for (int j = i + 1; j < num; j++) {
                    for (int k = 0; k < 3; ++k) {
                        if (values[i*max_channels+k] > values[j*max_channels+k]) {
                            desc_ptr[dcount1 / 8] |= (1 << (dcount1 % 8));
                        }
                        dcount1++;
                    }
                }
            }
        }
    }
    return desc;
}

Mat MyAkazeDescTest::run(vector<vector<Mat>> scale_space_y_arr2, vector<KeyPoint> key_points) {
    vector<vector<Mat>> scale_space_Ix_arr2, scale_space_Iy_arr2;
    GetScaleSpaceIxIy(scale_space_y_arr2, scale_space_Ix_arr2, scale_space_Iy_arr2);

    Mat desc = Get_Upright_MLDB_Full_Descriptor(scale_space_y_arr2, scale_space_Ix_arr2, scale_space_Iy_arr2, key_points);

    return desc;
}
