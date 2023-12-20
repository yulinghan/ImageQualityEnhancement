#include "brisk_feature_descriptors.hpp"

MyBriskFeatureDescriptorsTest::MyBriskFeatureDescriptorsTest(float patternScale) {
    vector<float> rList;
	std::vector<int> nList;

	// this is the standard pattern found to be suitable also
	rList.resize(5);
	nList.resize(5);
	double f=0.85*patternScale;

	rList[0]=f*0;
	rList[1]=f*2.9;
	rList[2]=f*4.9;
	rList[3]=f*7.4;
	rList[4]=f*10.8;

	nList[0]=1;
	nList[1]=10;
	nList[2]=14;
	nList[3]=15;
	nList[4]=20;

	GenerateKernel(rList, nList, 5.85*patternScale, 8.2*patternScale);
}

MyBriskFeatureDescriptorsTest::~MyBriskFeatureDescriptorsTest() {
}

void MyBriskFeatureDescriptorsTest::GenerateKernel(vector<float> &radiusList, vector<int> &numberList, float dMax, float dMin) {
    vector<int> indexChange;
    dMax_ = dMax;
    dMin_ = dMin;

    // get the total number of points
    int rings = (int)radiusList.size();
    points_ = 0; // remember the total number of points
    double sineThetaLookupTable[n_rot_];
    double cosThetaLookupTable[n_rot_];
    for(int ring = 0; ring < rings; ring++) {
        points_ += numberList[ring];
    }

    double cosval = 1., sinval = 0.;
    double dcos = cos(2*CV_PI/double(n_rot_)), dsin = sin(2*CV_PI/double(n_rot_));
    for(size_t rot = 0; rot < n_rot_; ++rot) {
        sineThetaLookupTable[rot] = sinval;
        cosThetaLookupTable[rot] = cosval;
        double t = sinval*dcos + cosval*dsin;
        cosval = cosval*dcos - sinval*dsin;
        sinval = t;
    }

    // set up the patterns
    patternPoints_ = new BriskPatternPoint[points_ * scales_ * n_rot_];

    // define the scale discretization:
    float lb_scale = (float)(std::log(scalerange_) / std::log(2.0));
    float lb_scale_step = lb_scale / (scales_);

    scaleList_ = new float[scales_];
    sizeList_ = new int[scales_];

    float sigma_scale = 1.3f;

    for(unsigned int scale = 0; scale < scales_; ++scale) {
        scaleList_[scale] = (float)std::pow((double) 2.0, (double) (scale * lb_scale_step));

        sizeList_[scale] = 0;
        BriskPatternPoint *patternIteratorOuter = patternPoints_ + (scale * n_rot_ * points_);

        // generate the pattern points look-up
        for (int ring = 0; ring < rings; ++ring) {
            double scaleRadiusProduct = scaleList_[scale] * radiusList[ring];
            float patternSigma = 0.0f;
            if (ring == 0) {
                patternSigma = sigma_scale * scaleList_[scale] * 0.5f;
            } else {
                patternSigma = (float) (sigma_scale * scaleList_[scale] * (double(radiusList[ring]))
                        * sin(CV_PI / numberList[ring]));
            }

            // adapt the sizeList if necessary
            const unsigned int size = cvCeil(((scaleList_[scale] * radiusList[ring]) + patternSigma)) + 1;
            if (sizeList_[scale] < size) {
                sizeList_[scale] = size;
            }

            for (int num = 0; num < numberList[ring]; ++num) {
                BriskPatternPoint *patternIterator = patternIteratorOuter;
                double alpha = (double(num)) * 2 * CV_PI / double(numberList[ring]);
                double sine_alpha = sin(alpha);
                double cosine_alpha = cos(alpha);

                for (size_t rot = 0; rot < n_rot_; ++rot) {
                    double cosine_theta = cosThetaLookupTable[rot];
                    double sine_theta = sineThetaLookupTable[rot];
    
                    patternIterator->x = (float) (scaleRadiusProduct *
                            (cosine_theta * cosine_alpha -
                             sine_theta * sine_alpha)); // feature rotation plus angle of the point
                    patternIterator->y = (float) (scaleRadiusProduct *
                            (sine_theta * cosine_alpha + cosine_theta * sine_alpha));
                    patternIterator->sigma = patternSigma;
                    patternIterator += points_;
                }
                ++patternIteratorOuter;
            }
        }
    }

    // now also generate pairings
    shortPairs_ = new BriskShortPair[points_ * (points_ - 1) / 2];
    longPairs_ = new BriskLongPair[points_ * (points_ - 1) / 2];
    noShortPairs_ = 0;
    noLongPairs_ = 0;

    // fill indexChange with 0..n if empty
    unsigned int indSize = (unsigned int)indexChange.size();
    if(indSize == 0) {
        indexChange.resize(points_ * (points_ - 1) / 2);
        indSize = (unsigned int)indexChange.size();

        for (unsigned int i = 0; i < indSize; i++) {
            indexChange[i] = i;
        }
    }
    float dMin_sq = dMin_ * dMin_;
    float dMax_sq = dMax_ * dMax_;
    for(int i = 1; i < points_; i++) {
        for (int j = 0; j < i; j++) { 
            //(find all the pairs)
            // point pair distance:
            const float dx = patternPoints_[j].x - patternPoints_[i].x;
            const float dy = patternPoints_[j].y - patternPoints_[i].y;
            const float norm_sq = (dx * dx + dy * dy);
            if (norm_sq > dMin_sq) {
                // save to long pairs
                BriskLongPair& longPair = longPairs_[noLongPairs_];
                longPair.weighted_dx = int((dx / (norm_sq)) * 2048.0 + 0.5);
                longPair.weighted_dy = int((dy / (norm_sq)) * 2048.0 + 0.5);
                longPair.i = i;
                longPair.j = j;
                ++noLongPairs_;
            } else if (norm_sq < dMax_sq) {
                // make sure the user passes something sensible
                BriskShortPair& shortPair = shortPairs_[indexChange[noShortPairs_]];
                shortPair.j = j;
                shortPair.i = i;
                ++noShortPairs_;
            }
        }
    }

    strings_ = (int) ceil((float(noShortPairs_)) / 128.0) * 4 * 4;
}

int MyBriskFeatureDescriptorsTest::smoothedIntensity(Mat& image, Mat& integral, float key_x, float key_y, int scale, int rot, int point) {

    // get the float position
    BriskPatternPoint& briskPoint = patternPoints_[scale * n_rot_ * points_ + rot * points_ + point];
    float xf = briskPoint.x + key_x;
    float yf = briskPoint.y + key_y;
    int x = int(xf);
    int y = int(yf);
    int& imagecols = image.cols;

    // get the sigma:
    float sigma_half = briskPoint.sigma;
    float area = 4.0f * sigma_half * sigma_half;

    // calculate output:
    int ret_val;
    if (sigma_half < 0.5) {
        //interpolation multipliers:
        int r_x = (int)((xf - x) * 1024);
        int r_y = (int)((yf - y) * 1024);
        int r_x_1 = (1024 - r_x);
        int r_y_1 = (1024 - r_y);
        uchar* ptr = &image.at<uchar>(y, x);
        size_t step = image.step;
        // just interpolate:
        ret_val = r_x_1 * r_y_1 * ptr[0] + r_x * r_y_1 * ptr[1] +
            r_x * r_y * ptr[step] + r_x_1 * r_y * ptr[step+1];

        return (ret_val + 512) / 1024;
    }

    // this is the standard case (simple, not speed optimized yet):
    // scaling:
    int scaling = (int)(4194304.0 / area);
    int scaling2 = int(float(scaling) * area / 1024.0);

    // the integral image is larger:
    int integralcols = imagecols + 1;

    // calculate borders
    float x_1 = xf - sigma_half;
    float x1 = xf + sigma_half;
    float y_1 = yf - sigma_half;
    float y1 = yf + sigma_half;

    int x_left = int(x_1 + 0.5);
    int y_top = int(y_1 + 0.5);
    int x_right = int(x1 + 0.5);
    int y_bottom = int(y1 + 0.5);

    // overlap area - multiplication factors:
    float r_x_1 = float(x_left) - x_1 + 0.5f;
    float r_y_1 = float(y_top) - y_1 + 0.5f;
    float r_x1 = x1 - float(x_right) + 0.5f;
    float r_y1 = y1 - float(y_bottom) + 0.5f;
    int dx = x_right - x_left - 1;
    int dy = y_bottom - y_top - 1;
    int A = (int)((r_x_1 * r_y_1) * scaling);
    int B = (int)((r_x1 * r_y_1) * scaling);
    int C = (int)((r_x1 * r_y1) * scaling);
    int D = (int)((r_x_1 * r_y1) * scaling);
    int r_x_1_i = (int)(r_x_1 * scaling);
    int r_y_1_i = (int)(r_y_1 * scaling);
    int r_x1_i = (int)(r_x1 * scaling);
    int r_y1_i = (int)(r_y1 * scaling);

    if (dx + dy > 2) {
        // now the calculation:
        const uchar* ptr = image.ptr() + x_left + imagecols * y_top;
        // first the corners:
        ret_val = A * int(*ptr);
        ptr += dx + 1;
        ret_val += B * int(*ptr);
        ptr += dy * imagecols + 1;
        ret_val += C * int(*ptr);
        ptr -= dx + 1;
        ret_val += D * int(*ptr);

        // next the edges:
        int* ptr_integral = integral.ptr<int>() + x_left + integralcols * y_top + 1;
        // find a simple path through the different surface corners
        int tmp1 = (*ptr_integral);
        ptr_integral += dx;
        int tmp2 = (*ptr_integral);
        ptr_integral += integralcols;
        int tmp3 = (*ptr_integral);
        ptr_integral++;
        int tmp4 = (*ptr_integral);
        ptr_integral += dy * integralcols;
        int tmp5 = (*ptr_integral);
        ptr_integral--;
        int tmp6 = (*ptr_integral);
        ptr_integral += integralcols;
        int tmp7 = (*ptr_integral);
        ptr_integral -= dx;
        int tmp8 = (*ptr_integral);
        ptr_integral -= integralcols;
        int tmp9 = (*ptr_integral);
        ptr_integral--;
        int tmp10 = (*ptr_integral);
        ptr_integral -= dy * integralcols;
        int tmp11 = (*ptr_integral);
        ptr_integral++;
        int tmp12 = (*ptr_integral);

        // assign the weighted surface integrals:
        int upper = (tmp3 - tmp2 + tmp1 - tmp12) * r_y_1_i;
        int middle = (tmp6 - tmp3 + tmp12 - tmp9) * scaling;
        int left = (tmp9 - tmp12 + tmp11 - tmp10) * r_x_1_i;
        int right = (tmp5 - tmp4 + tmp3 - tmp6) * r_x1_i;
        int bottom = (tmp7 - tmp6 + tmp9 - tmp8) * r_y1_i;

        return (ret_val + upper + middle + left + right + bottom + scaling2 / 2) / scaling2;
    }

    // now the calculation:
    uchar* ptr = image.ptr() + x_left + imagecols * y_top;
    // first row:
    ret_val = A * int(*ptr);
    ptr++;
    uchar* end1 = ptr + dx;
    for (; ptr < end1; ptr++) {
        ret_val += r_y_1_i * int(*ptr);
    }
    ret_val += B * int(*ptr);
    // middle ones:
    ptr += imagecols - dx - 1;
    uchar* end_j = ptr + dy * imagecols;
    for (; ptr < end_j; ptr += imagecols - dx - 1) {
        ret_val += r_x_1_i * int(*ptr);
        ptr++;
        uchar* end2 = ptr + dx;
        for (; ptr < end2; ptr++) {
            ret_val += int(*ptr) * scaling;
        }
        ret_val += r_x1_i * int(*ptr);
    }
    // last row:
    ret_val += D * int(*ptr);
    ptr++;
    uchar* end3 = ptr + dx;
    for (; ptr < end3; ptr++) {
        ret_val += r_y1_i * int(*ptr);
    }
    ret_val += C * int(*ptr);

    return (ret_val + scaling2 / 2) / scaling2;
}

Mat MyBriskFeatureDescriptorsTest::run(Mat src, vector<KeyPoint> key_points) {
    vector<int> kscales;
    float basicSize06 = basicSize_ * 0.6f;
    float log2 = 0.693147180559945f;
    float lb_scalerange = (float)(std::log(scalerange_) / (log2));

    for(int i=0; i<key_points.size(); i++) {
        int scale = std::max((int)(scales_ / lb_scalerange * (std::log(key_points[i].size / (basicSize06)) / log2) + 0.5), 0);
        kscales.push_back(scale);
    }

    Mat _integral; // the integral image
    integral(src, _integral);

    Mat descriptors = Mat::zeros(Size((int)key_points.size(), strings_), CV_8U);
    uchar* ptr = descriptors.ptr();

    int _values[points_];
    int t1;
    int t2;
    for(int k=0; k<key_points.size(); k++) {
        KeyPoint& kp = key_points[k];
        int& scale = kscales[k];
        float& x = kp.pt.x;
        float& y = kp.pt.y;

        for (unsigned int i = 0; i < points_; i++) {
            _values[i] = smoothedIntensity(src, _integral, x, y, scale, 0, i);
        }

        int direction0 = 0;
        int direction1 = 0;
        // now iterate through the long pairings
        BriskLongPair* max = longPairs_ + noLongPairs_;
        for (BriskLongPair* iter = longPairs_; iter < max; ++iter) {
            t1 = *(_values + iter->i);
            t2 = *(_values + iter->j);
            int delta_t = (t1 - t2);
            // update the direction:
            const int tmp0 = delta_t * (iter->weighted_dx) / 1024;
            const int tmp1 = delta_t * (iter->weighted_dy) / 1024;
            direction0 += tmp0;
            direction1 += tmp1;
        }
        kp.angle = (float)(atan2((float) direction1, (float) direction0) / CV_PI * 180.0);

        int theta;
        if (kp.angle==-1) {
            // don't compute the gradient direction, just assign a rotation of 0
            theta = 0;
        } else {
            theta = (int) (n_rot_ * (kp.angle / (360.0)) + 0.5);
            if (theta < 0) {
                theta += n_rot_;
            }
            if (theta >= int(n_rot_)) {
                theta -= n_rot_;
            }
        }

        if (kp.angle < 0) {
            kp.angle += 360.f;
        }

        int shifter = 0;
        for (int i = 0; i < points_; i++) {
            _values[i] = smoothedIntensity(src, _integral, x, y, scale, theta, i);
        }

        // now iterate through all the pairings
        unsigned int* ptr2 = (unsigned int*)ptr;
        BriskShortPair* max_short = shortPairs_ + noShortPairs_;
        for (BriskShortPair* iter = shortPairs_; iter < max_short; ++iter) {
            t1 = *(_values + iter->i);
            t2 = *(_values + iter->j);
            if (t1 > t2) {
                *ptr2 |= ((1) << shifter);
            }
            ++shifter;
            if (shifter == 32) {
                shifter = 0;
                ++ptr2;
            }
        }
        ptr += strings_;
    }

    return descriptors;
}
