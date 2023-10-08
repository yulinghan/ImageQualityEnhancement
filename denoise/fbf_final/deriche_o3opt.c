/*
 * Copyright (c) 2016, Anmol Popli <anmol.ap020@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @file deriche_o3opt.c
 * @brief Fast gaussian filter using deriche IIR approximation of O(3)
 *
 * @author ANMOL POPLI <anmol.ap020@gmail.com>
 *         PRAVIN NAIR  <sreehari1390@gmail.com>
 **/

#include "headersreq.h"
void convolve_deriche2D(int rows, int columns, int sigma, double complex** ip_padded);
double Nc[3],Dc[3],Na[3],Da[3],scale;
int w;
//double filter[w+1];
/**
 * \brief Convolve input array with 1D Causal filter
 *        (Deriche Recursive algorithm) 
 * \param in        Pointer to input array
 * \param out       Pointer to output array
 * \param datasize  Input array size
 *
 * This routine performs constant time convolution of the
 * 1D input array of complex doubles with 1D Causal filter
 * of Deriche Recursive algorithm. The 1D filter is an
 * IIR filter. 
 */
void convolve_dericheCausal(double complex* in, double complex* out, int datasize,double *filter) {
    
    int i, j;
    
    /* Compute first 3 output elements non-recursively */
    out[w]=0;
    for(i=0; i<w+1; i++)
    {
        out[w] += in[i]*filter[i];
    }
    out[w+1]=0;
    for(i=0; i<w+1; i++)
    {
        out[w+1] += in[i+1]*filter[i];
    }
    out[w+2]=0;
    for(i=0; i<w+1; i++)
    {
        out[w+2] += in[i+2]*filter[i];
    }
    
    /* Recursive computation of output in forward direction using filter parameters Nc, Dc and scale */
    for (i=w+3; i<datasize-w; i++) {
        out[i] = 0;
        for (j=0; j<3; j++) {
            out[i] += Nc[j]*in[i-(2-j)]/scale;
            out[i] -= Dc[j]*out[i-(3-j)];
        }
    }
         
}

/**
 * \brief Convolve input array with 1D AntiCausal filter
 *        (Deriche Recursive algorithm) 
 * \param in        Pointer to input array
 * \param out       Pointer to output array
 * \param datasize  Input array size
 *
 * This routine performs constant time convolution of the
 * 1D input array of complex doubles with 1D AntiCausal filter
 * of Deriche Recursive algorithm. The 1D filter is an
 * IIR filter. 
 */
void convolve_dericheAnticausal(double complex* in, double complex* out, int datasize,double *filter) {
    
    int i, j;
    
    /* Compute last 3 output elements non-recursively */
    out[datasize-1-w]=0;
    for(i=0; i<w; i++)
    {
        out[datasize-1-w] += in[datasize-1-i]*filter[i];
    }
    out[datasize-2-w]=0;
    for(i=0; i<w; i++)
    {
        out[datasize-2-w] += in[datasize-2-i]*filter[i];
    }
    out[datasize-3-w]=0;
    for(i=0; i<w; i++)
    {
        out[datasize-3-w] += in[datasize-3-i]*filter[i];
    }
    
    /* Recursive computation of output in backward direction using filter parameters Na, Da and scale */
    for (i=datasize-4-w; i>=w; i--) {
        out[i] = 0;
        for (j=0; j<3; j++) {
            out[i] += Na[j]*in[i+(j+1)]/scale;
            out[i] -= Da[j]*out[i+(j+1)];
        }
    }
    
}

/**
 * \brief Convolve input array with 1D Gaussian filter
 *        (Deriche Recursive algorithm) 
 * \param in        Pointer to input array
 * \param out       Pointer to output array
 * \param datasize  Input array size
 *
 * This routine performs constant time convolution of the
 * 1D input array of complex doubles with 1D Gaussian filter
 * using Deriche Recursive algorithm. The input array is separately
 * convolved with Causal and AntiCausal filters and the results are
 * added to obtain the output array.
 */
void convolve_deriche1D(double complex* in, double complex* out, int datasize,double *filter) {
    /** \brief Array to store output of Causal filter convolution */
    double complex out_causal[datasize];
    convolve_dericheCausal(in, out_causal, datasize,filter);
    convolve_dericheAnticausal(in, out, datasize,filter);
    int i;
    for (i=0; i<datasize; i++)  in[i] = out_causal[i] + out[i];
}

/**
 * \brief Apply 2D Gaussian filter to input image
 *        (Deriche Recursive Algorithm) 
 * \param rows      Image height
 * \param columns   Image width
 * \param sigma     Gaussian kernel standard deviation
 * \param ip_padded Pointer to input image
 * \param op_padded Pointer to output image
 *
 * This routine applies 2D Gaussian filter of s.d.
 * sigma to input image ip_padded of dimensions
 * rows x columns and computes output image op_padded.
 * 1D filter is first convolved along rows and then
 * along columns. The 1D convolution is performed using
 * Deriche's fast recursive algorithm.
 */
void convolve_deriche2D(int rows, int columns, int sigma, double complex** ip_padded) {
    
    /** \brief Filter radius */
    w = 3*sigma;
    /** \brief Array to store filter weights */
    double *filter=calloc(w+1,sizeof(double));
    /** \brief Impulse response parameters  */
    double a0 = -0.8929, a1 = 1.021, b0 = 1.512, w0 = 1.475, c0 = 1.898, b1 = 1.556;
    
    /** \brief Transfer function coefficients */
    double n22c = c0*exp(-2*b0/sigma) + (a0*cos(w0/sigma)-a1*sin(w0/sigma))*exp(-(b0+b1)/sigma);
    double n11c = -((a0*cos(w0/sigma)-a1*sin(w0/sigma))*exp(-b0/sigma) + a0*exp(-b1/sigma) + 2*c0*exp(-b0/sigma)*cos(w0/sigma));
    double n00c = a0 + c0;
    double d33c = -exp(-(2*b0+b1)/sigma);
    double d22c = exp(-2*b0/sigma) + 2*exp(-(b0+b1)/sigma)*cos(w0/sigma);
    double d11c = -(2*exp(-b0/sigma)*cos(w0/sigma) + exp(-b1/sigma));
    double d11a = d11c, d22a = d22c, d33a = d33c;
    double n33a = -d33c*n00c;
    double n22a = n22c - d22c*n00c;
    double n11a = n11c - d11c*n00c;
    Nc[0]= n22c; Nc[1]= n11c; Nc[2]=n00c;
        Dc[0]= d33c; Dc[1]= d22c; Dc[2]=d11c;
        Na[0]= n11a; Na[1]= n22a; Na[2]=n33a;
        Da[0]= d11a; Da[1]= d22a; Da[2]=d33a;   
    /** \brief Scale to normalize filter weights */
    scale = (Nc[0]+Nc[1]+Nc[2])/(1+Dc[0]+Dc[1]+Dc[2]) + (Na[0]+Na[1]+Na[2])/(1+Da[0]+Da[1]+Da[2]);
    
    /* Compute normalized filter weights */
    int i,j,k;
    for (i=0;i<w+1;i++)
    {
        double gnum = -(i-w)*(i-w);
        double gden = 2*sigma*sigma;
        filter[i] = exp(gnum/gden)/scale;
    }
    
    /* Symmetric padding of input image with padding width equal to the filter radius w */
    symmetric_padding(rows,columns,ip_padded,w);
    
    /* Convolve each row with 1D Gaussian filter */
    double complex *out_t = calloc(columns+(2*w),sizeof(double complex));
    for (i=0; i<rows+2*w; i++) convolve_deriche1D(ip_padded[i], out_t, columns+2*w,filter);
    free(out_t);
        double complex *intemp=calloc(rows+(2*w),sizeof(double complex)),*outtemp=calloc(rows+(2*w),sizeof(double complex));
        for (j=w; j<columns+w; j++)
        {
            /* Convolve each column with 1D Gaussian filter */
            for (i=0;i<rows+(2*w);i++) intemp[i]=ip_padded[i][j];
            convolve_deriche1D(intemp, outtemp, rows+2*w,filter);
            /* Store the convolved column in row of output matrix*/
            for (i=0;i<rows+(2*w);i++) ip_padded[i][j]=intemp[i];
        }
        free(filter);
        free(intemp);
        free(outtemp);
}

