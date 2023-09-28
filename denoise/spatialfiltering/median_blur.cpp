#include "median_blur.hpp"

MyMedianBlurTest::MyMedianBlurTest() {
}

MyMedianBlurTest::~MyMedianBlurTest() {
}

Mat MyMedianBlurTest::Run(Mat src, int r) {
    Mat out = Mat::zeros(src.size(), src.type());
    int ksize = 2*r+1;
    
    for (int i=r; i<src.rows-r; i++) {  
        for (int j=r; j<src.cols-r; j++) {

            vector<uchar> window;
            for (int ii=i-r; ii<=i+r; ii++) {
                for (int jj=j-r; jj<=j+r; jj++) {
                    window.push_back(src.at<uchar>(ii, jj));
                }
            }

            for (int m = 0; m <= ksize*ksize/2; m++) {  
                int min = m;  
                for (int n=m+1; n<ksize*ksize; n++) {
                    if (window[n] < window[min]) { 
                        min = n;
                    }
                }
                int temp = window[m];  
                window[m] = window[min];  
                window[min] = temp;  
            }  
            out.at<uchar>(i, j) = window[window.size()/2];
        }  
    }  

	return out;
}
