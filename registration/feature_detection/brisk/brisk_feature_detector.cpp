#include "brisk_feature_detector.hpp"
#include "orb_keypoint.hpp"

MyBriskFeatureDetectorTest::MyBriskFeatureDetectorTest() {
}

MyBriskFeatureDetectorTest::~MyBriskFeatureDetectorTest() {
}

vector<Mat> MyBriskFeatureDetectorTest::ConstructPyramid(Mat src, int layers) {
    vector<Mat> src_pyr;

    src_pyr.push_back(src.clone());

    Mat cur_mat;
    resize(src, cur_mat, Size(2*src.cols/3, 2*src.rows/3));
    src_pyr.push_back(cur_mat);

    for(int i=2; i<layers; i+=2) {   
        resize(src, cur_mat, src_pyr[i-2].size()/2);
        src_pyr.push_back(cur_mat);
        resize(src, cur_mat, src_pyr[i-1].size()/2);
        src_pyr.push_back(cur_mat);
    }

    return src_pyr;
}

int MyBriskFeatureDetectorTest::GetScore(Mat src, int pixel[25], int threshold, int x, int y) {
    MyOrbKeyPointTest *my_orb_point_test = new MyOrbKeyPointTest();
    const uchar* ptr = src.ptr<uchar>(y) + x;
    int score = my_orb_point_test->CornerScore(ptr, pixel, threshold);

    return score;
}

float MyBriskFeatureDetectorTest::subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2,
									const int s_1_0, const int s_1_1, const int s_1_2,
									const int s_2_0, const int s_2_1, const int s_2_2,
									float& delta_x, float& delta_y){

	// the coefficients of the 2d quadratic function least-squares fit:
	register int tmp1 =        s_0_0 + s_0_2 - 2*s_1_1 + s_2_0 + s_2_2;
    register int coeff1 = 3*(tmp1 + s_0_1 - ((s_1_0 + s_1_2)<<1) + s_2_1);
    register int coeff2 = 3*(tmp1 - ((s_0_1+ s_2_1)<<1) + s_1_0 + s_1_2 );
    register int tmp2 =                                  s_0_2 - s_2_0;
    register int tmp3 =                         (s_0_0 + tmp2 - s_2_2);
    register int tmp4 =                                   tmp3 -2*tmp2;
    register int coeff3 =                    -3*(tmp3 + s_0_1 - s_2_1);
	register int coeff4 =                    -3*(tmp4 + s_1_0 - s_1_2);
	register int coeff5 =            (s_0_0 - s_0_2 - s_2_0 + s_2_2)<<2;
	register int coeff6 = -(s_0_0  + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1)<<1) - 5*s_1_1  + s_2_0  + s_2_2)<<1;


	// 2nd derivative test:
	register int H_det=4*coeff1*coeff2 - coeff5*coeff5;

	if(H_det==0){
		delta_x=0.0;
		delta_y=0.0;
		return float(coeff6)/18.0;
	}

	if(!(H_det>0&&coeff1<0)){
		// The maximum must be at the one of the 4 patch corners.
		int tmp_max=coeff3+coeff4+coeff5;
		delta_x=1.0; delta_y=1.0;

		int tmp = -coeff3+coeff4-coeff5;
		if(tmp>tmp_max){
			tmp_max=tmp;
			delta_x=-1.0; delta_y=1.0;
		}
		tmp = coeff3-coeff4-coeff5;
		if(tmp>tmp_max){
			tmp_max=tmp;
			delta_x=1.0; delta_y=-1.0;
		}
		tmp = -coeff3-coeff4+coeff5;
		if(tmp>tmp_max){
			tmp_max=tmp;
			delta_x=-1.0; delta_y=-1.0;
		}
		return float(tmp_max+coeff1+coeff2+coeff6)/18.0;
	}

	// this is hopefully the normal outcome of the Hessian test
	delta_x=float(2*coeff2*coeff3 - coeff4*coeff5)/float(-H_det);
	delta_y=float(2*coeff1*coeff4 - coeff3*coeff5)/float(-H_det);
	// TODO: this is not correct, but easy, so perform a real boundary maximum search:
	bool tx=false; bool tx_=false; bool ty=false; bool ty_=false;
	if(delta_x>1.0) tx=true;
	else if(delta_x<-1.0) tx_=true;
	if(delta_y>1.0) ty=true;
	if(delta_y<-1.0) ty_=true;

	if(tx||tx_||ty||ty_){
		// get two candidates:
		float delta_x1=0.0, delta_x2=0.0, delta_y1=0.0, delta_y2=0.0;
		if(tx) {
			delta_x1=1.0;
			delta_y1=-float(coeff4+coeff5)/float(2*coeff2);
			if(delta_y1>1.0) delta_y1=1.0; else if (delta_y1<-1.0) delta_y1=-1.0;
		}
		else if(tx_) {
			delta_x1=-1.0;
			delta_y1=-float(coeff4-coeff5)/float(2*coeff2);
			if(delta_y1>1.0) delta_y1=1.0; else if (delta_y1<-1.0) delta_y1=-1.0;
		}
		if(ty) {
			delta_y2=1.0;
			delta_x2=-float(coeff3+coeff5)/float(2*coeff1);
			if(delta_x2>1.0) delta_x2=1.0; else if (delta_x2<-1.0) delta_x2=-1.0;
		}
		else if(ty_) {
			delta_y2=-1.0;
			delta_x2=-float(coeff3-coeff5)/float(2*coeff1);
			if(delta_x2>1.0) delta_x2=1.0; else if (delta_x2<-1.0) delta_x2=-1.0;
		}
		// insert both options for evaluation which to pick
		float max1 = (coeff1*delta_x1*delta_x1+coeff2*delta_y1*delta_y1
				+coeff3*delta_x1+coeff4*delta_y1
				+coeff5*delta_x1*delta_y1
				+coeff6)/18.0;
		float max2 = (coeff1*delta_x2*delta_x2+coeff2*delta_y2*delta_y2
				+coeff3*delta_x2+coeff4*delta_y2
				+coeff5*delta_x2*delta_y2
				+coeff6)/18.0;
		if(max1>max2) {
			delta_x=delta_x1;
			delta_y=delta_x1;
			return max1;
		}
		else{
			delta_x=delta_x2;
			delta_y=delta_x2;
			return max2;
		}
	}

	// this is the case of the maximum inside the boundaries:
	return (coeff1*delta_x*delta_x+coeff2*delta_y*delta_y
			+coeff3*delta_x+coeff4*delta_y
			+coeff5*delta_x*delta_y
			+coeff6)/18.0;
}

float MyBriskFeatureDetectorTest::GetScoreMaxAbove(vector<Mat> src_pyr, int pixel[25], int layer, int x_layer, int y_layer, int threshold, int center, 
                                    bool& ismax, float& dx, float& dy) {
    // relevant floating point coordinates
    ismax = false;
    float x_1, x1, y_1, y1;

    // the layer above
    Mat layerAbove = src_pyr[layer + 1];

    if (layer % 2 == 0) {
        // octave
        x_1 = float(4 * (x_layer) - 1 - 2) / 6.0f;
        x1 = float(4 * (x_layer) - 1 + 2) / 6.0f;
        y_1 = float(4 * (y_layer) - 1 - 2) / 6.0f;
        y1 = float(4 * (y_layer) - 1 + 2) / 6.0f;
    } else {
        // intra
        x_1 = float(6 * (x_layer) - 1 - 3) / 8.0f;
        x1 = float(6 * (x_layer) - 1 + 3) / 8.0f;
        y_1 = float(6 * (y_layer) - 1 - 3) / 8.0f;
        y1 = float(6 * (y_layer) - 1 + 3) / 8.0f;
    }

    // check the first row
    int max_x = (int)x_1 + 1;
    int max_y = (int)y_1 + 1;
    float tmp_max;
    float maxval = (float)GetScore(layerAbove, pixel, threshold, x_1, y_1);
    if (maxval > center) {
        return 0;
    }
    for (int x = (int)x_1 + 1; x <= int(x1); x++) {
        tmp_max = (float)GetScore(layerAbove, pixel, threshold, x, y_1);
        if (tmp_max > center) {
            return 0;
        }
        if (tmp_max > maxval) {
            maxval = tmp_max;
            max_x = x;
        }
    }
    tmp_max = (float)GetScore(layerAbove, pixel, threshold, x1, y_1);
    if (tmp_max > center) {
        return 0;
    }
    if (tmp_max > maxval) {
        maxval = tmp_max;
        max_x = int(x1);
    }

    // middle rows
    for (int y = (int)y_1 + 1; y <= int(y1); y++) {
        tmp_max = (float)GetScore(layerAbove, pixel, threshold, x_1, y);
        if (tmp_max > center) {
            return 0;
        }
        if (tmp_max > maxval) {
            maxval = tmp_max;
            max_x = int(x_1 + 1);
            max_y = y;
        }
        for (int x = (int)x_1 + 1; x <= int(x1); x++) {
            tmp_max = (float)GetScore(layerAbove, pixel, threshold, x, y);
            if (tmp_max > center) {
                return 0;
            }
            if (tmp_max > maxval) {
                maxval = tmp_max;
                max_x = x;
                max_y = y;
            }
        }
        tmp_max = (float)GetScore(layerAbove, pixel, threshold, x1, y);
        if (tmp_max > center) {
            return 0;
        }
        if (tmp_max > maxval) {
            maxval = tmp_max;
            max_x = int(x1);
            max_y = y;
        }
    }

    // bottom row
    tmp_max = (float)GetScore(layerAbove, pixel, threshold, x_1, y1);
    if (tmp_max > maxval) {
        maxval = tmp_max;
        max_x = int(x_1 + 1);
        max_y = int(y1);
    }
    for (int x = (int)x_1 + 1; x <= int(x1); x++) {
        tmp_max = (float)GetScore(layerAbove, pixel, threshold, x, y1);
        if (tmp_max > maxval) {
            maxval = tmp_max;
            max_x = x;
            max_y = int(y1);
        }
    }
    tmp_max = (float)GetScore(layerAbove, pixel, threshold, x1, y1);
    if (tmp_max > maxval) {
        maxval = tmp_max;
        max_x = int(x1);
        max_y = int(y1);
    }

    //find dx/dy:
    int s_0_0 = GetScore(layerAbove, pixel, threshold, max_x - 1, max_y - 1);
    int s_1_0 = GetScore(layerAbove, pixel, threshold, max_x, max_y - 1);
    int s_2_0 = GetScore(layerAbove, pixel, threshold, max_x + 1, max_y - 1);
    int s_2_1 = GetScore(layerAbove, pixel, threshold, max_x + 1, max_y);
    int s_1_1 = GetScore(layerAbove, pixel, threshold, max_x, max_y);
    int s_0_1 = GetScore(layerAbove, pixel, threshold, max_x - 1, max_y);
    int s_0_2 = GetScore(layerAbove, pixel, threshold, max_x - 1, max_y + 1);
    int s_1_2 = GetScore(layerAbove, pixel, threshold, max_x, max_y + 1);
    int s_2_2 = GetScore(layerAbove, pixel, threshold, max_x + 1, max_y + 1);

    float dx_1, dy_1;
    float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, dx_1, dy_1);

    // calculate dx/dy in above coordinates
    float real_x = float(max_x) + dx_1;
    float real_y = float(max_y) + dy_1;
    bool returnrefined = true;
    if (layer % 2 == 0) {
        dx = (real_x * 6.0f + 1.0f) / 4.0f - float(x_layer);
        dy = (real_y * 6.0f + 1.0f) / 4.0f - float(y_layer);
    } else {
        dx = (real_x * 8.0f + 1.0f) / 6.0f - float(x_layer);
        dy = (real_y * 8.0f + 1.0f) / 6.0f - float(y_layer);
    }

    // saturate
    if (dx > 1.0f) {
        dx = 1.0f;
        returnrefined = false;
    }
    if (dx < -1.0f) {
        dx = -1.0f;
        returnrefined = false;
    }
    if (dy > 1.0f) {
        dy = 1.0f;
        returnrefined = false;
    }
    if (dy < -1.0f) {
        dy = -1.0f;
        returnrefined = false;
    }

    // done and ok.
    ismax = true;
    if (returnrefined) {
        return std::max(refined_max, maxval);
    }

    return maxval;
}

float MyBriskFeatureDetectorTest::GetScoreMaxBelow(vector<Mat> src_pyr, int pixel[25], int layer, int x_layer, int y_layer, int threshold, int center, 
                                    bool& ismax, float& dx, float& dy) {
    ismax = false;
    // relevant floating point coordinates
    float x_1, x1, y_1, y1;

    if (layer % 2 == 0) {
        // octave
        x_1 = float(8 * (x_layer) + 1 - 4) / 6.0f;
        x1 = float(8 * (x_layer) + 1 + 4) / 6.0f;
        y_1 = float(8 * (y_layer) + 1 - 4) / 6.0f;
        y1 = float(8 * (y_layer) + 1 + 4) / 6.0f;
    } else {
        x_1 = float(6 * (x_layer) + 1 - 3) / 4.0f;
        x1 = float(6 * (x_layer) + 1 + 3) / 4.0f;
        y_1 = float(6 * (y_layer) + 1 - 3) / 4.0f;
        y1 = float(6 * (y_layer) + 1 + 3) / 4.0f;
    }

    // the layer below
    Mat layerBelow = src_pyr[layer - 1];

    // check the first row
    int max_x = (int)x_1 + 1;
    int max_y = (int)y_1 + 1;
    float tmp_max;
    float max = (float)GetScore(layerBelow, pixel, threshold, x_1, y_1);
    if(max > threshold) {
        return 0;
    }

    for(int x = (int)x_1 + 1; x <= int(x1); x++) {
        tmp_max = (float)GetScore(layerBelow, pixel, threshold, x, y_1);
        if (tmp_max > threshold) {
            return 0;
        }
        if (tmp_max > max) {
            max = tmp_max;
            max_x = x;
        }
    }
    tmp_max = (float)GetScore(layerBelow, pixel, threshold, x1, y_1);
    if (tmp_max > threshold) {
        return 0;
    }
    if (tmp_max > max) {
        max = tmp_max;
        max_x = int(x1);
    }

    // middle rows
    for (int y = (int)y_1 + 1; y <= int(y1); y++) {
        tmp_max = (float)GetScore(layerBelow, pixel, threshold, x_1, y);
        if (tmp_max > threshold) {
            return 0;
        }
        if (tmp_max > max) {
            max = tmp_max;
            max_x = int(x_1 + 1);
            max_y = y;
        }
        for (int x = (int)x_1 + 1; x <= int(x1); x++) {
            tmp_max = (float)GetScore(layerBelow, pixel, threshold, x, y);
            if (tmp_max > threshold) {
                return 0;
            }
            if (tmp_max == max) {
                const int t1 = 2 * (GetScore(layerBelow, pixel, threshold, x-1, y) + GetScore(layerBelow, pixel, threshold, x+1, y)
                            + GetScore(layerBelow, pixel, threshold, x, y+1) + GetScore(layerBelow, pixel, threshold, x, y-1))
                            + (GetScore(layerBelow, pixel, threshold, x+1, y+1) + GetScore(layerBelow, pixel, threshold, x-1, y+1)
                            + GetScore(layerBelow, pixel, threshold, x+1, y-1) + GetScore(layerBelow, pixel, threshold, x-1, y-1));

                const int t2 = 2 * (GetScore(layerBelow, pixel, threshold, max_x-1, max_y) + GetScore(layerBelow, pixel, threshold, max_x+1, max_y)
                            + GetScore(layerBelow, pixel, threshold, max_x, max_y+1) + GetScore(layerBelow, pixel, threshold, max_x, max_y-1))
                            + (GetScore(layerBelow, pixel, threshold, max_x+1, max_y+1) + GetScore(layerBelow, pixel, threshold, max_x-1, max_y+1)
                            + GetScore(layerBelow, pixel, threshold, max_x+1, max_y-1) + GetScore(layerBelow, pixel, threshold, max_x-1, max_y-1));

                if (t1 > t2) {
                    max_x = x;
                    max_y = y;
                }
            }
            if (tmp_max > max) {
                max = tmp_max;
                max_x = x;
                max_y = y;
            }
        }
        tmp_max = (float)GetScore(layerBelow, pixel, threshold, x1, y);
        if (tmp_max > threshold)
            return 0;
        if (tmp_max > max)
        {
            max = tmp_max;
            max_x = int(x1);
            max_y = y;
        }
    }

    // bottom row
    tmp_max = (float)GetScore(layerBelow, pixel, threshold, x_1, y1);
    if (tmp_max > max) {
        max = tmp_max;
        max_x = int(x_1 + 1);
        max_y = int(y1);
    }
    for (int x = (int)x_1 + 1; x <= int(x1); x++) {
        tmp_max = (float)GetScore(layerBelow, pixel, threshold, x, y1);
        if (tmp_max > max) {
            max = tmp_max;
            max_x = x;
            max_y = int(y1);
        }
    }
    tmp_max = (float)GetScore(layerBelow, pixel, threshold, x1, y1);
    if (tmp_max > max) {
        max = tmp_max;
        max_x = int(x1);
        max_y = int(y1);
    }

    //find dx/dy:
    int s_0_0 = GetScore(layerBelow, pixel, threshold, max_x - 1, max_y - 1);
    int s_1_0 = GetScore(layerBelow, pixel, threshold, max_x, max_y - 1);
    int s_2_0 = GetScore(layerBelow, pixel, threshold, max_x + 1, max_y - 1);
    int s_2_1 = GetScore(layerBelow, pixel, threshold, max_x + 1, max_y);
    int s_1_1 = GetScore(layerBelow, pixel, threshold, max_x, max_y);
    int s_0_1 = GetScore(layerBelow, pixel, threshold, max_x - 1, max_y);
    int s_0_2 = GetScore(layerBelow, pixel, threshold, max_x - 1, max_y + 1);
    int s_1_2 = GetScore(layerBelow, pixel, threshold, max_x, max_y + 1);
    int s_2_2 = GetScore(layerBelow, pixel, threshold, max_x + 1, max_y + 1);

    float dx_1, dy_1;
    float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, dx_1, dy_1);

    // calculate dx/dy in above coordinates
    float real_x = float(max_x) + dx_1;
    float real_y = float(max_y) + dy_1;
    bool returnrefined = true;
    if (layer % 2 == 0) {
        dx = (float)((real_x * 6.0 + 1.0) / 8.0) - float(x_layer);
        dy = (float)((real_y * 6.0 + 1.0) / 8.0) - float(y_layer);
    } else {
        dx = (float)((real_x * 4.0 - 1.0) / 6.0) - float(x_layer);
        dy = (float)((real_y * 4.0 - 1.0) / 6.0) - float(y_layer);
    }

    // saturate
    if (dx > 1.0) {
        dx = 1.0f;
        returnrefined = false;
    }
    if (dx < -1.0f) {
        dx = -1.0f;
        returnrefined = false;
    }
    if (dy > 1.0f) {
        dy = 1.0f;
        returnrefined = false;
    }
    if (dy < -1.0f) {
        dy = -1.0f;
        returnrefined = false;
    }

    // done and ok.
    ismax = true;
    if (returnrefined) {
        return std::max(refined_max, max);
    }
    return max;
}

float MyBriskFeatureDetectorTest::refine1D(const float s_05, const float s0, const float s05, float& max) {
    int i_05 = int(1024.0 * s_05 + 0.5);
    int i0 = int(1024.0 * s0 + 0.5);
    int i05 = int(1024.0 * s05 + 0.5);

    int three_a = 16 * i_05 - 24 * i0 + 8 * i05;
    // second derivative must be negative:
    if (three_a >= 0) {
        if (s0 >= s_05 && s0 >= s05) {
            max = s0;
            return 1.0f;
        }
        if (s_05 >= s0 && s_05 >= s05) {
            max = s_05;
            return 0.75f;
        }
        if (s05 >= s0 && s05 >= s_05) {
            max = s05;
            return 1.5f;
        }
    }

    int three_b = -40 * i_05 + 54 * i0 - 14 * i05;
    // calculate max location:
    float ret_val = -float(three_b) / float(2 * three_a);
    // saturate and return
    if (ret_val < 0.75) {
        ret_val = 0.75;
    } else if (ret_val > 1.5) {
        ret_val = 1.5; // allow to be slightly off bounds ...?
    }

    int three_c = +24 * i_05 - 27 * i0 + 6 * i05;
    max = float(three_c) + float(three_a) * ret_val * ret_val + float(three_b) * ret_val;
    max /= 3072.0f;

    return ret_val;
}

float MyBriskFeatureDetectorTest::refine1D_1(const float s_05, const float s0, const float s05, float& max) {
    int i_05 = int(1024.0 * s_05 + 0.5);
    int i0 = int(1024.0 * s0 + 0.5);
    int i05 = int(1024.0 * s05 + 0.5);

    //  4.5000   -9.0000    4.5000
    //-10.5000   18.0000   -7.5000
    //  6.0000   -8.0000    3.0000

    int two_a = 9 * i_05 - 18 * i0 + 9 * i05;
    // second derivative must be negative:
    if (two_a >= 0) {
        if (s0 >= s_05 && s0 >= s05) {
            max = s0;
            return 1.0f;
        }
        if (s_05 >= s0 && s_05 >= s05) {
            max = s_05;
            return 0.6666666666666666666666666667f;
        }
        if (s05 >= s0 && s05 >= s_05) {
            max = s05;
            return 1.3333333333333333333333333333f;
        }
    }

    int two_b = -21 * i_05 + 36 * i0 - 15 * i05;
    // calculate max location:
    float ret_val = -float(two_b) / float(2 * two_a);
    // saturate and return
    if (ret_val < 0.6666666666666666666666666667f)
        ret_val = 0.666666666666666666666666667f;
    else if (ret_val > 1.33333333333333333333333333f)
        ret_val = 1.333333333333333333333333333f;
    int two_c = +12 * i_05 - 16 * i0 + 6 * i05;
    max = float(two_c) + float(two_a) * ret_val * ret_val + float(two_b) * ret_val;
    max /= 2048.0f;
    return ret_val;
}

vector<KeyPoint> MyBriskFeatureDetectorTest::GetKeyPoints(vector<Mat> src_pyr, int threshold, int layers) {
    vector<vector<KeyPoint>> fast_keypoint_arr;
    vector<KeyPoint> brisk_keypoint;
    vector<float> scale_arr;
    vector<float> offset_arr;
    int patternSize = 16;
    int basicSize = 12;
    float scale;

    for(int i=0; i<layers; i++) {
        MyOrbKeyPointTest *my_orb_point_test = new MyOrbKeyPointTest();
        vector<KeyPoint> corners = my_orb_point_test->Run(src_pyr[i], threshold);
        fast_keypoint_arr.push_back(corners);
        float cur_scale  = (float)src_pyr[0].rows / (float)src_pyr[i].rows;
        float cur_offset = 0.5f * cur_scale - 0.5f;
        scale_arr.push_back(cur_scale);
        offset_arr.push_back(cur_offset);
    }

    for(int i=1; i<layers-1; i++) {
        int key_num =  fast_keypoint_arr[i].size();
        float cur_scale = scale_arr[i];
        float cur_offset= offset_arr[i];

        int pixel[25];
        MyOrbKeyPointTest *my_orb_point_test = new MyOrbKeyPointTest();
        my_orb_point_test->MakeOffsets(pixel, (int)src_pyr[i].step, patternSize);

        for(int j=0; j<key_num; j++) {
            bool ismax = true;
            int x_layer = (int)fast_keypoint_arr[i][j].pt.x;
            int y_layer = (int)fast_keypoint_arr[i][j].pt.y;
            int center = GetScore(src_pyr[i], pixel, threshold, x_layer, y_layer);

            float delta_x_above = 0, delta_y_above = 0;
            float max_above = GetScoreMaxAbove(src_pyr, pixel, i, x_layer, y_layer, 
                                threshold, center, ismax, delta_x_above, delta_y_above);

            if(!ismax) {
                continue;
            }
            
            float max;
            float delta_x_below, delta_y_below;
            float max_below_float;
            int max_below = 0;
            float x, y;
            if(i % 2 == 0) {
                max_below_float = GetScoreMaxBelow(src_pyr, pixel, i, x_layer, y_layer,
                                 threshold, center, ismax, delta_x_below, delta_y_below);
                if(!ismax) {
                    continue;
                }

                // get the patch on this layer:
                int s_0_0 = GetScore(src_pyr[i], pixel, threshold, x_layer-1, y_layer-1);
                int s_1_0 = GetScore(src_pyr[i], pixel, threshold, x_layer, y_layer-1);
                int s_2_0 = GetScore(src_pyr[i], pixel, threshold, x_layer+1, y_layer-1);
                int s_2_1 = GetScore(src_pyr[i], pixel, threshold, x_layer+1, y_layer);
                int s_1_1 = GetScore(src_pyr[i], pixel, threshold, x_layer, y_layer);
                int s_0_1 = GetScore(src_pyr[i], pixel, threshold, x_layer-1, y_layer);
                int s_0_2 = GetScore(src_pyr[i], pixel, threshold, x_layer-1, y_layer+1);
                int s_1_2 = GetScore(src_pyr[i], pixel, threshold, x_layer, y_layer+1);
                int s_2_2 = GetScore(src_pyr[i], pixel, threshold, x_layer+1, y_layer+1);

                float delta_x_layer, delta_y_layer;
                float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_layer,
                        delta_y_layer);

                // calculate the relative scale (1D maximum):
                scale = refine1D(max_below_float, std::max(float(center), max_layer), max_above, max);
                if (scale > 1.0) {
                    // interpolate the position:
                    const float r0 = (1.5f - scale) / .5f;
                    const float r1 = 1.0f - r0;
                    x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer)) * cur_scale + cur_offset;
                    y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer)) * cur_scale + cur_offset;
                } else {
                    // interpolate the position:
                    const float r0 = (scale - 0.75f) / 0.25f;
                    const float r_1 = 1.0f - r0;
                    x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer)) * cur_scale + cur_offset;
                    y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer)) * cur_scale + cur_offset;
                }
            } else {
                // on intra
                // check the patch below:
                max_below = GetScoreMaxBelow(src_pyr, pixel, i, x_layer, y_layer,
                                 threshold, center, ismax, delta_x_below, delta_y_below);
                if (!ismax) {
                    continue;
                }

                // get the patch on this layer:
                int s_0_0 = GetScore(src_pyr[i], pixel, threshold, x_layer-1, y_layer-1);
                int s_1_0 = GetScore(src_pyr[i], pixel, threshold, x_layer, y_layer-1);
                int s_2_0 = GetScore(src_pyr[i], pixel, threshold, x_layer+1, y_layer-1);
                int s_2_1 = GetScore(src_pyr[i], pixel, threshold, x_layer+1, y_layer);
                int s_1_1 = GetScore(src_pyr[i], pixel, threshold, x_layer, y_layer);
                int s_0_1 = GetScore(src_pyr[i], pixel, threshold, x_layer-1, y_layer);
                int s_0_2 = GetScore(src_pyr[i], pixel, threshold, x_layer-1, y_layer+1);
                int s_1_2 = GetScore(src_pyr[i], pixel, threshold, x_layer, y_layer+1);
                int s_2_2 = GetScore(src_pyr[i], pixel, threshold, x_layer+1, y_layer+1);

                float delta_x_layer, delta_y_layer;
                float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_layer,
                        delta_y_layer);

                // calculate the relative scale (1D maximum):
                scale = refine1D_1(max_below, std::max(float(center), max_layer), max_above, max);
                if (scale > 1.0) {
                    // interpolate the position:
                    const float r0 = 4.0f - scale * 3.0f;
                    const float r1 = 1.0f - r0;
                    x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer)) * cur_scale + cur_offset;
                    y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer)) * cur_scale + cur_offset;
                } else {
                    // interpolate the position:
                    const float r0 = scale * 3.0f - 2.0f;
                    const float r_1 = 1.0f - r0;
                    x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer)) * cur_scale + cur_offset;
                    y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer)) * cur_scale + cur_offset;
                }
            }
            brisk_keypoint.push_back(KeyPoint(x, y, basicSize * scale, -1, max, i));
        }
    }

    return brisk_keypoint;
}

vector<KeyPoint> MyBriskFeatureDetectorTest::run(Mat src, int threshold, int octaves) {
    int layers = octaves * 2;
    vector<Mat> src_pyr = ConstructPyramid(src, layers);
    vector<KeyPoint> corners = GetKeyPoints(src_pyr, threshold, layers);

    return corners;
}
