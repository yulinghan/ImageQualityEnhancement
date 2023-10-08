#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>
#include "qx_recursive_bilateral_filter.h"

using namespace std;
using namespace cv;

#define QX_DEF_PADDING					10

inline double** qx_allocd(int r,int c,int padding=QX_DEF_PADDING)
{
	double *a,**p;
	a=(double*) malloc(sizeof(double)*(r*c+padding));
	if(a==NULL) {printf("qx_allocd() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(double**) malloc(sizeof(double*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}

inline double *** qx_allocd_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
	double *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(double*) malloc(sizeof(double)*(n*rc+padding));
	if(a==NULL) {printf("qx_allocd_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(double**) malloc(sizeof(double*)*n*r);
    pp=(double***) malloc(sizeof(double**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}

inline unsigned char *** qx_allocu_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
	unsigned char *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(unsigned char*) malloc(sizeof(unsigned char )*(n*rc+padding));
	if(a==NULL) {printf("qx_allocu_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(unsigned char**) malloc(sizeof(unsigned char*)*n*r);
    pp=(unsigned char***) malloc(sizeof(unsigned char**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}

int example_filtering(int argc,char*argv[])
{
	float sigma_spatial=atof(argv[3]);
	float sigma_range=atof(argv[4]);

	Mat src = imread(argv[1]);
	imwrite(argv[1], src);
	int h = src.rows;
	int w = src.cols;

	unsigned char***texture=qx_allocu_3(h,w,3);
	double***image=qx_allocd_3(h,w,3);
	double***image_filtered=qx_allocd_3(h,w,3);
	double***temp=qx_allocd_3(h,w,3);
	double***temp_2=qx_allocd_3(2,w,3);

	for(int y=0;y<h;y++) {
		for(int x=0;x<w;x++) {
			for(int c=0;c<3;c++) {
				image[y][x][c] = src.at<uchar>(y, x*3+c);
				texture[y][x][c] = src.at<uchar>(y, x*3+c);
			}
		}
	}

	int nr_iteration=10;
	double**temp_factor=qx_allocd(h*2+2,w);
	for(int i=0;i<nr_iteration;i++) {
		qx_recursive_bilateral_filter(image_filtered,image,texture,sigma_spatial,sigma_range,h,w,temp,temp_2,temp_factor,&(temp_factor[h]),&(temp_factor[h+h]));
	}

	Mat out = src.clone();
	for(int y=0;y<h;y++) {
		for(int x=0;x<w;x++) {
			for(int c=0;c<3;c++) {
				out.at<uchar>(y, x*3+c) = (uchar)image_filtered[y][x][c];
			}
		}
	}
	imshow(argv[2], out);
	waitKey(0);

	return(0);
}

int main(int argc,char*argv[]) {
	return(example_filtering(argc,argv));
}
