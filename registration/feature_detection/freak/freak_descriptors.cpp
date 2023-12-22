#include "freak_descriptors.hpp"
#include <bitset>

MyFreakDescriptorsTest::MyFreakDescriptorsTest() {
    patternScale = 22.0f;
    nOctaves = 4;

    BuildPattern();
}

MyFreakDescriptorsTest::~MyFreakDescriptorsTest() {
}

void MyFreakDescriptorsTest::BuildPattern() {
    patternLookup = new PatternPoint[NB_SCALES * NB_ORIENTATION * NB_POINTS];
    double scaleStep = std::pow(2.0, (double)(nOctaves)/NB_SCALES); // 2 ^ ( (nOctaves-1) /nbScales)
    double scalingFactor, alpha, beta, theta = 0;

    // pattern definition, radius normalized to 1.0 (outer point position+sigma=1.0)
    int n[8] = {6,6,6,6,6,6,6,1}; // number of points on each concentric circle (from outer to inner)
    double bigR(2.0/3.0); // bigger radius
    double smallR(2.0/24.0); // smaller radius
    double unitSpace( (bigR-smallR)/21.0 ); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)
    // radii of the concentric cirles (from outer to inner)
    double radius[8] = {bigR, bigR-6*unitSpace, bigR-11*unitSpace, bigR-15*unitSpace, bigR-18*unitSpace, bigR-20*unitSpace, smallR, 0.0};
    // sigma of pattern points (each group of 6 points on a concentric cirle has the same sigma)
    double sigma[8] = {radius[0]/2.0, radius[1]/2.0, radius[2]/2.0,
                        radius[3]/2.0, radius[4]/2.0, radius[5]/2.0,
                        radius[6]/2.0, radius[6]/2.0};

    max_border = 0;
    max_sigma = 0;
    // fill the lookup table
    for(int scaleIdx=0; scaleIdx < NB_SCALES; ++scaleIdx) {
        patternSizes[scaleIdx] = 0; // proper initialization
        scalingFactor = std::pow(scaleStep,scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

        for(int orientationIdx = 0; orientationIdx < NB_ORIENTATION; ++orientationIdx) {
            theta = double(orientationIdx)* 2*CV_PI/double(NB_ORIENTATION); // orientation of the pattern
            int pointIdx = 0;

            PatternPoint* patternLookupPtr = &patternLookup[0];
            for(size_t i = 0; i < 8; ++i) {
                for(int k = 0 ; k < n[i]; ++k) {
                    beta = CV_PI/n[i] * (i%2); // orientation offset so that groups of points on each circles are staggered
                    alpha = double(k)* 2*CV_PI/double(n[i])+beta+theta;

                    // add the point to the look-up table
                    PatternPoint& point = patternLookupPtr[scaleIdx*NB_ORIENTATION * NB_POINTS + orientationIdx * NB_POINTS + pointIdx ];
                    point.x = static_cast<float>(radius[i] * cos(alpha) * scalingFactor * patternScale);
                    point.y = static_cast<float>(radius[i] * sin(alpha) * scalingFactor * patternScale);
                    point.sigma = static_cast<float>(sigma[i] * scalingFactor * patternScale);
        
                    if(max_border < abs(point.x)) {
                        max_border = abs(point.x);
                    }
                    if(max_border < abs(point.y)) {
                        max_border = abs(point.y);
                    }
                    if(max_sigma < point.sigma) {
                        max_sigma = point.sigma;
                    }
                    // adapt the sizeList if necessary
                    const int sizeMax = static_cast<int>(ceil((radius[i]+sigma[i])*scalingFactor*patternScale)) + 1;
                    if(patternSizes[scaleIdx] < sizeMax) {
                        patternSizes[scaleIdx] = sizeMax;
                    }
                    ++pointIdx;
                }
            }
        }
    }

    // build the list of orientation pairs
    orientationPairs[0].i=0; orientationPairs[0].j=3; orientationPairs[1].i=1; orientationPairs[1].j=4; orientationPairs[2].i=2; orientationPairs[2].j=5;
    orientationPairs[3].i=0; orientationPairs[3].j=2; orientationPairs[4].i=1; orientationPairs[4].j=3; orientationPairs[5].i=2; orientationPairs[5].j=4;
    orientationPairs[6].i=3; orientationPairs[6].j=5; orientationPairs[7].i=4; orientationPairs[7].j=0; orientationPairs[8].i=5; orientationPairs[8].j=1;

    orientationPairs[9].i=6; orientationPairs[9].j=9; orientationPairs[10].i=7; orientationPairs[10].j=10; orientationPairs[11].i=8; orientationPairs[11].j=11;
    orientationPairs[12].i=6; orientationPairs[12].j=8; orientationPairs[13].i=7; orientationPairs[13].j=9; orientationPairs[14].i=8; orientationPairs[14].j=10;
    orientationPairs[15].i=9; orientationPairs[15].j=11; orientationPairs[16].i=10; orientationPairs[16].j=6; orientationPairs[17].i=11; orientationPairs[17].j=7;

    orientationPairs[18].i=12; orientationPairs[18].j=15; orientationPairs[19].i=13; orientationPairs[19].j=16; orientationPairs[20].i=14; orientationPairs[20].j=17;
    orientationPairs[21].i=12; orientationPairs[21].j=14; orientationPairs[22].i=13; orientationPairs[22].j=15; orientationPairs[23].i=14; orientationPairs[23].j=16;
    orientationPairs[24].i=15; orientationPairs[24].j=17; orientationPairs[25].i=16; orientationPairs[25].j=12; orientationPairs[26].i=17; orientationPairs[26].j=13;

    orientationPairs[27].i=18; orientationPairs[27].j=21; orientationPairs[28].i=19; orientationPairs[28].j=22; orientationPairs[29].i=20; orientationPairs[29].j=23;
    orientationPairs[30].i=18; orientationPairs[30].j=20; orientationPairs[31].i=19; orientationPairs[31].j=21; orientationPairs[32].i=20; orientationPairs[32].j=22;
    orientationPairs[33].i=21; orientationPairs[33].j=23; orientationPairs[34].i=22; orientationPairs[34].j=18; orientationPairs[35].i=23; orientationPairs[35].j=19;

    orientationPairs[36].i=24; orientationPairs[36].j=27; orientationPairs[37].i=25; orientationPairs[37].j=28; orientationPairs[38].i=26; orientationPairs[38].j=29;
    orientationPairs[39].i=30; orientationPairs[39].j=33; orientationPairs[40].i=31; orientationPairs[40].j=34; orientationPairs[41].i=32; orientationPairs[41].j=35;
    orientationPairs[42].i=36; orientationPairs[42].j=39; orientationPairs[43].i=37; orientationPairs[43].j=40; orientationPairs[44].i=38; orientationPairs[44].j=41;

    for(unsigned m = NB_ORIENPAIRS; m--; ) {
        float dx = patternLookup[orientationPairs[m].i].x-patternLookup[orientationPairs[m].j].x;
        float dy = patternLookup[orientationPairs[m].i].y-patternLookup[orientationPairs[m].j].y;
        float norm_sq = (dx*dx+dy*dy);
        orientationPairs[m].weight_dx = cvRound((dx/(norm_sq))*4096.0);
        orientationPairs[m].weight_dy = cvRound((dy/(norm_sq))*4096.0);
    }

    // build the list of description pairs
    vector<DescriptionPair> allPairs;
    for(int i = 1; i < NB_POINTS; ++i) {
        for(int j = 0; j < i; ++j) {
            DescriptionPair pair = {i, j};
            allPairs.push_back(pair);
        }
    }

    for(int i = 0; i < NB_PAIRS; ++i ) {
        descriptionPairs[i] = allPairs[DEF_PAIRS[i]];
    }
}

void MyFreakDescriptorsTest::extractDescriptor(int *pointsValue, void ** ptr) {
    bitset<NB_PAIRS>** ptrScalar = (bitset<NB_PAIRS>**) ptr;

    // extracting descriptor preserving the order of SSE version
    int cnt = 0;
    for( int n = 7; n < NB_PAIRS; n += 128) {
        for( int m = 8; m--; ) {
            int nm = n-m;
            for(int kk = nm+15*8; kk >= nm; kk-=8, ++cnt) {
                (*ptrScalar)->set(kk, pointsValue[descriptionPairs[cnt].i] >= pointsValue[descriptionPairs[cnt].j]);
            }
        }
    }
    --(*ptrScalar);
}


Mat MyFreakDescriptorsTest::ComputeDescriptors(Mat image, vector<Point> keypoints){
    Mat imgIntegral;
    integral(image, imgIntegral);
    vector<Point>::iterator kpBegin = keypoints.begin(); // used in std::vector erase function
    float sizeCst = static_cast<float>(NB_SCALES/(LOG2* nOctaves));
    int pointsValue[NB_POINTS];
    int thetaIdx = 0;
    int direction0;
    int direction1;

    for( size_t k = keypoints.size(); k--; ) {
        int pattern = max_border + max_sigma;
        if(keypoints[k].x <= pattern ||
                keypoints[k].y <= pattern ||
                keypoints[k].x >= image.cols - pattern ||
                keypoints[k].y >= image.rows - pattern) {
            keypoints.erase(kpBegin+k);
        }
    }

    Mat descriptors;
    // allocate descriptor memory, estimate orientations, extract descriptors
    if(!extAll) {
        // extract the best comparisons only
        descriptors = Mat::zeros(Size(NB_PAIRS/8, (int)keypoints.size()), CV_8U);
        uchar *ptr = descriptors.ptr(0);

        for( size_t k = keypoints.size(); k--; ) {
            // estimate orientation (gradient)
            //每个特征的43个取值点作均值滤波
            for(int i = NB_POINTS; i--; ) {
                pointsValue[i] = MeanIntensity(image, imgIntegral, keypoints[k].x, keypoints[k].y, 0, 0, i);
            }
            direction0 = 0;
            direction1 = 0;
            //对长距离类的45个取值点对，计算特征主方向
            for(int m = 45; m--; ) {
                //iterate through the orientation pairs
                const int delta = (pointsValue[orientationPairs[m].i]-pointsValue[orientationPairs[m].j]);
                direction0 += delta*(orientationPairs[m].weight_dx)/2048;
                direction1 += delta*(orientationPairs[m].weight_dy)/2048;
            }

            float angle = static_cast<float>(atan2((float)direction1,(float)direction0)*(180.0/CV_PI));//estimate orientation
            thetaIdx = cvRound(NB_ORIENTATION*angle*(1/360.0));

            if(thetaIdx < 0) {
                thetaIdx += NB_ORIENTATION;
            }
            if(thetaIdx >= NB_ORIENTATION) {
                thetaIdx -= NB_ORIENTATION;
            }

            // extract descriptor at the computed orientation
            //特征43个取值点，坐标根据主方向旋转后，作均值滤波
            for( int i = NB_POINTS; i--; ) {
                pointsValue[i] = MeanIntensity(image, imgIntegral, keypoints[k].x, keypoints[k].y, 0, thetaIdx, i);
            }

            //对预设的512个取值点对进行数据编码，得到64个uchar类似的特征描述
            for( int n = 0; n < NB_PAIRS/8; n ++) {
                int shifter = 0;
                for( int m=0; m<8; m++) {
                    int cnt = n*8 + m;
                    int t1 = pointsValue[descriptionPairs[cnt].i];
                    int t2 = pointsValue[descriptionPairs[cnt].j];
                    if (t1 > t2) {
                        *ptr |= ((1) << shifter);
                    }
                    ++shifter;
                }
                ptr++;
            }
        }
    } else {// extract all possible comparisons for selection
        descriptors = Mat::zeros(Size(128, (int)keypoints.size()), CV_8U);

        for(size_t k = 0; k<keypoints.size(); k++) {
            uchar *ptr = descriptors.ptr((int)k);

            //特征点主方向计算
            for(int i = NB_POINTS;i--; ) {
                pointsValue[i] = MeanIntensity(image, imgIntegral, keypoints[k].x,keypoints[k].y, 0, 0, i);
            }

            direction0 = 0;
            direction1 = 0;
            for( int m = 45; m--; ) {
                //iterate through the orientation pairs
                const int delta = (pointsValue[ orientationPairs[m].i ]-pointsValue[ orientationPairs[m].j ]);
                direction0 += delta*(orientationPairs[m].weight_dx)/2048;
                direction1 += delta*(orientationPairs[m].weight_dy)/2048;
            }

            float angle = static_cast<float>(atan2((float)direction1,(float)direction0)*(180.0/CV_PI)); //estimate orientation
            thetaIdx = cvRound(NB_ORIENTATION*angle*(1/360.0));

            if(thetaIdx < 0) {
                thetaIdx += NB_ORIENTATION;
            }
            if(thetaIdx >= NB_ORIENTATION) {
                thetaIdx -= NB_ORIENTATION;
            }

            //特征43个采样点主方向旋转后，计算采样点均值滤波
            for(int i = NB_POINTS; i--; ) {
                pointsValue[i] = MeanIntensity(image, imgIntegral, keypoints[k].x, keypoints[k].y, 0, thetaIdx, i);
            }

            int shifter = 0;
            for(int i=0; i<NB_POINTS; i++) {
                for(int j=0; j<i; j++) {
                        int t1 = pointsValue[i];
                        int t2 = pointsValue[j];
                        if(shifter == 8) {
                            shifter = 0;
                            ptr++;
                        }
                        if (t1 > t2) {
                            *ptr |= ((1) << shifter);
                        }
                        shifter += 1;
                }
            }
        }
    }

    return descriptors;
}

int MyFreakDescriptorsTest::MeanIntensity(Mat image, Mat integral, float kp_x, float kp_y,
                        int scale, int rot, int point) {
    PatternPoint& FreakPoint = patternLookup[scale*NB_ORIENTATION*NB_POINTS + rot*NB_POINTS + point];
    float xf = FreakPoint.x+kp_x;
    float yf = FreakPoint.y+kp_y;
    int x = int(xf);
    int y = int(yf);

    // get the sigma:
    float radius = FreakPoint.sigma;

    // calculate output:
    if(radius < 0.5) {
        // interpolation multipliers:
        int r_x = static_cast<int>((xf-x)*1024);
        int r_y = static_cast<int>((yf-y)*1024);
        int r_x_1 = (1024-r_x);
        int r_y_1 = (1024-r_y);

        int ret_val;
        // linear interpolation:
        ret_val = r_x_1*r_y_1*int(image.at<int>(y  , x  ))
                + r_x  *r_y_1*int(image.at<int>(y  , x+1))
                + r_x_1*r_y  *int(image.at<int>(y+1, x  ))
                + r_x  *r_y  *int(image.at<int>(y+1, x+1));
        //return the rounded mean
        ret_val += 2 * 1024 * 1024;
        return ret_val / (4 * 1024 * 1024);
    }

    // expected case:
    // calculate borders
    int x_left = cvRound(xf-radius);
    int y_top = cvRound(yf-radius);
    int x_right = cvRound(xf+radius+1);//integral image is 1px wider
    int y_bottom = cvRound(yf+radius+1);//integral image is 1px higher

    int ret_val;
    ret_val = integral.at<int>(y_bottom,x_right);//bottom right corner
    ret_val -= integral.at<int>(y_bottom,x_left);
    ret_val += integral.at<int>(y_top,x_left);
    ret_val -= integral.at<int>(y_top,x_right);
    int area = (x_right - x_left) * (y_bottom - y_top);
    ret_val = (ret_val + area/2) / area;

    return ret_val;
}

Mat MyFreakDescriptorsTest::SelectPairs(Mat src, vector<Point>& key_points, double corrTresh) {
    extAll = true;

    // compute descriptors with all pairs
    Mat descriptors = ComputeDescriptors(src, key_points);

    //descriptor in floating point format (each bit is a float)
    Mat descriptorsFloat = Mat::zeros(descriptors.rows, 903, CV_32F);

    Mat new_descriptors = Mat::zeros(descriptors.rows, NB_PAIRS/8, CV_8U);;

    bitset<1024>* ptr = (bitset<1024>*) (descriptors.data+(descriptors.rows-1)*descriptors.step[0]);
    for(int m = descriptors.rows; m--; ) {
        for( int n = 903; n--; ) {
            if( ptr->test(n) == true ) {
                descriptorsFloat.at<float>(m,n)=1.0f;
            }
        }
        --ptr;
    }

    vector<PairStat> pairStat;
    for(int n = 903; n--; ) {
        // the higher the variance, the better --> mean = 0.5
        PairStat tmp = { fabs( mean(descriptorsFloat.col(n))[0]-0.5 ) ,n};
        pairStat.push_back(tmp);
    }

    sort(pairStat.begin(),pairStat.end(), sortMean());

    vector<PairStat> bestPairs;
    for(int m = 0; m < 903; ++m ) {
        double corrMax(0);

        for(size_t n = 0; n < bestPairs.size(); ++n) {
            int idxA = bestPairs[n].idx;
            int idxB = pairStat[m].idx;
            double corr(0);
            // compute correlation between 2 pairs
            //直方图比较每列特征的相关性，如果新的这列和已选中的列相关性太高，表示以选中列中有类似特征表现，当前列抛弃。
            //最终从903列中，保留最不相关的512列，进行特征降维度
            corr = fabs(compareHist(descriptorsFloat.col(idxA), descriptorsFloat.col(idxB), HISTCMP_CORREL));

            if(corr > corrMax) {
                corrMax = corr;
                if(corrMax >= corrTresh) {
                    break;
                }
            }
        }

        if(corrMax < corrTresh) {
            bestPairs.push_back(pairStat[m]);
        }
        if(bestPairs.size() >= NB_PAIRS) {
            break;
        }
    }

    for(int i=0; i<new_descriptors.rows; i++) {
        for(int j=0; j<NB_PAIRS/8; j++) {
            new_descriptors.at<uchar>(i, j) = descriptors.at<uchar>(i, bestPairs[i].idx);
        }
    }

    return new_descriptors;
}

Mat MyFreakDescriptorsTest::run(Mat src, vector<Point> key_points) {

    //903组采样点，随机选512组生成特征描述
    Mat descriptors1 = ComputeDescriptors(src, key_points);

    //903组采样点直接生成特征描述。
    //相关性检测，去掉比较重复的特征描述维度，最终特征描述从903降维到512组
    double corrTresh = 0.8;
    Mat descriptors2 = SelectPairs(src, key_points, corrTresh);

    return descriptors2;
}
