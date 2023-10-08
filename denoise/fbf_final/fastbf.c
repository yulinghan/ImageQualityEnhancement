/*
 * Copyright (c) 2016, Pravin Nair <sreehari1390@gmail.com>
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
 * @file fastbf.c
 * @brief fast bilateral filter and required functions
 *
 * @author PRAVIN NAIR  <sreehari1390@gmail.com>
 **/

#include "headersreq.h"

int shiftableBF(int m, int n, int sigmas, double sigmar, double** img, double** outimg, int cores, program_params* params,double eps);

/**
 * \brief Calculate norm of vector
 * \param a         Pointer to vector
 * \param n         Length of vector
 * \return Norm
 *
 * This routine calculates norm of vector
 * of length n.
 */
double norm(double *a,int n)
{
    double m_sum = fabs(a[0]);
    int i;
    for ( i=1; i<n;++i){
        if (fabs(a[i])>m_sum) m_sum = fabs(a[i]);
    }
    return m_sum;
}

/**
 * \brief Apply fast shiftable bilateral filter to input image
 * \param m         Image height
 * \param n         Image width
 * \param sigmas    Standard deviation of spatial kernel
 * \param sigmar    Standard deviation of range kernel
 * \param img       Pointer to input image
 * \param outimg    Pointer to output image
 * \param cores     Number of physical cores on system
 * \param params    Pointer to Program parameters like
 *                  coefficients
 *
 * This routine applies the fast shiftable bilateral filter
 * with parameters sigmas & sigmar to input image img of
 * dimensions m x n and computes output image outimg.
 * The algorithm used is Fourier Basis approximation.
 * The Gaussian spatial convolutions are performed using
 * 'Deriche' or 'Young and van Vliet' fast recursive algorithms,
 * dpending on sigmar and maximum local dynamic range of the image.
 * The convolutions are performed parallelly with one thread
 * assigned for each physical core on the system.
 */
int shiftableBF(int m, int n, int sigmas, double sigmar, double** img, double** outimg, int cores, program_params* params,double eps) {
    int i,j,k,l;
    int w = 6*sigmas+1; /** \brief Filter width */
    int c = (w-1)/2; /** \brief Filter radius */

    /* Fourier Basis Algorithm */  
    double T=maxfilterfind(img,w,m,n); /* Finding maximum local dynamic range which is image independent */
    double Tmax = max(T,ceil(3.2*sigmar)); /* New half period of the filter */
    params->T = Tmax;    
    int K=(int)(2*Tmax+1); /* Period = 2*Tmax+1 */
    double a1=(1/(2*Tmax+1));
    double omegao=(2*M_PI)/(2*Tmax+1); 
    double* basis=(double*)calloc(K,sizeof(double)); /* Vector holding DFT basis components in each iteration (for every frequency components)*/
    double* kernel=(double*)calloc(K,sizeof(double)); /* Range kernel discretization vector in range [-Tmax,Tmax]*/
    double* approx=(double*)calloc(K,sizeof(double)); /* Approximation of range kernel using basis*/
    int Kapprox=1; /* Number of coefficients required to get approximation error less than eps*/
    double* coff=(double*)calloc(Tmax+1,sizeof(double)); /* DFT coefficients*/    
    double dftcoeff=0; /* Variable holding DFT coefficient in each iteration */
    /* Calculating DC coefficient*/
    for (i=0;i<K;i++)
    {
        basis[i]=1;
        kernel[i]=exp((-0.5*(i-Tmax)*(i-Tmax))/(sigmar*sigmar));
        dftcoeff += (a1*basis[i]*kernel[i]);
    }
    double approxerrorarr[K]; /* Array containing pointwise approximation error*/
    for (i=0;i<K;i++)
    {
	approx[i]=dftcoeff*basis[i];
        approxerrorarr[i]=kernel[i]-approx[i];
    }   
    coff[0]=dftcoeff;
    double approxerror=norm(approxerrorarr,K); /* Approximation error Linfinity norm*/

    /* Calculating AC coefficients till range kernel approximation is less than eps and total number of DFT coefficients used should be less than Tmax*/
    while ((approxerror>eps) && (Kapprox<=Tmax)){
	dftcoeff=0;
	Kapprox=Kapprox+1;
	for (i=0;i<K;i++)
	{
            basis[i]=cos((Kapprox-1)*omegao*(i-Tmax)); /* Calculating DFT basis */
            dftcoeff += (basis[i]*kernel[i]);
        }        
	/* Multiplication by a1 is to normalize the basis and multiplying by 2 is to take into account for negative frequency
         component as well . Multiplying by 2 is done here so that we can avoid some checks in case of parallelization code
         as this is required only for AC frequency components. This is done to avoid changes in later part of code.*/
	dftcoeff*=(2*a1);
	for (i=0;i<K;i++)
        {
            approx[i]+=(dftcoeff*basis[i]);
            approxerrorarr[i]=kernel[i]-approx[i];
        }
        coff[Kapprox-1]=dftcoeff;
        approxerror=norm(approxerrorarr,K);    
    }
    free(approx);
    free(basis);
    free(kernel);

    params->K = Kapprox;
    params->coeff = (double*) calloc(Kapprox, sizeof(double));
    for (i=0; i<Kapprox; i++)  params->coeff[i] = coff[i];
    /* End of algorithm for finding appropripriate number of DFT coefficients for range kernel approximation */
    /*******************************************************************************************************************/

    /* Computation of Filtered image */
    double **P=alloc_array(m,n); /** \brief Matrix to store unnormalized filtered image */
    double **Q=alloc_array(m,n); /** \brief Matrix to store weight sums for normalization */

    double complex** F1 = alloc_array_complex(m+w-1,n+w-1); /** \brief Recursive parameter/basis matrix F1 for frequency omegao, required to compute Auxiliary images */
    for (i=0;i<m;i++)
    {
        for (j=0;j<n;j++)
        {
            F1[i+c][j+c]=exp(I*omegao*img[i][j]);
        }
    }
    #ifdef _OPENMP
        /* The auxiliary images are convolved with spatial Gaussian parallelly */
        int tid, procid;
        int chunk = (int) (Kapprox)/cores; /** \brief Number of iterations assigned to each thread at fork */
	if (chunk==0)
	    chunk=1; /* Auxillary image recursion is not required . So exp(I*omegao*k*img[i][j] has to be found for every k*/
        /* one thread per physical core */
        omp_set_num_threads(cores);
#pragma omp parallel shared(chunk,Kapprox,m,n,c,w,sigmas,P,Q,img,F1,coff,omegao) private(k,i,j,tid,procid)
    #endif
    {
        #ifdef _OPENMP
            tid = omp_get_thread_num();
            procid = sched_getcpu();
        #endif
        /** \brief Matrices for Auxiliary images */
        double complex **F=alloc_array_complex(m+w-1,n+w-1),**G=alloc_array_complex(m+w-1,n+w-1),**H=alloc_array_complex(m+w-1,n+w-1);
        #ifdef _OPENMP
            /** \brief P and Q private to thread */
            double ** P_k = alloc_array(m,n), **Q_k = alloc_array(m,n);
#pragma omp for schedule(static,chunk) nowait
        #endif
        for (k=0;k<Kapprox;k++)
        {
            /* Compute auxiliary images */
            #ifdef _OPENMP
                if (k%chunk==0) {
                    for (i=0;i<m;i++)
                    {
                        for (j=0;j<n;j++)
                        {
                            F[i+c][j+c]=exp(I*omegao*k*img[i][j]);
                            G[i+c][j+c]=conj(F[i+c][j+c]);
                            H[i+c][j+c]=G[i+c][j+c]*img[i][j];
                        }
                    }
            #else
                if (k==0) {
                    for (i=0;i<m;i++)
                    {
                        for (j=0;j<n;j++)
                        {
                            F[i+c][j+c]=1;
                            G[i+c][j+c]=1;
                            H[i+c][j+c]=G[i+c][j+c]*img[i][j];
                        }
                    }
            #endif
            } else {
                for (i=0;i<m;i++)
                {
                    for (j=0;j<n;j++)
                    {
                        F[i+c][j+c]=F[i+c][j+c]*F1[i+c][j+c];
                        G[i+c][j+c]=conj(F[i+c][j+c]);
                        H[i+c][j+c]=G[i+c][j+c]*img[i][j];
                    }
                }
            }
    
            /* Gaussian filter applied to auxiliary images, algo decided by ratio Tmax/sigmar */
            if ((Tmax/sigmar) < 3.5) {
                convolve_deriche2D(m,n,sigmas,H);  
                convolve_deriche2D(m,n,sigmas,G);

            } else {
                convolve_young2D(m,n,sigmas,H);    
                convolve_young2D(m,n,sigmas,G);
            }
            /* Update P and Q */
            #ifdef _OPENMP
                for (i=0;i<m;i++)
                {
                    for (j=0;j<n;j++)
                    {
                        P_k[i][j]+=creal(coff[k]*F[i+c][j+c]*H[i+c][j+c]);
                        Q_k[i][j]+=creal(coff[k]*F[i+c][j+c]*G[i+c][j+c]);
                    }
                }
        }

        /* Compute global P and Q from their private versions */
#pragma omp critical
        {
            for (i=0; i<m; i++) {
                for (j=0; j<n; j++) {
                    P[i][j] += P_k[i][j];
                    Q[i][j] += Q_k[i][j];
                }
            }
        }

        /* Deallocate thread private matrices */
        dealloc_array_fl(P_k,m);
        dealloc_array_fl(Q_k,m);
#else
            for (i=0;i<m;i++)
            {
                for (j=0;j<n;j++)
                {
                    P[i][j]+=creal(coff[k]*F[i+c][j+c]*H[i+c][j+c]);
                    Q[i][j]+=creal(coff[k]*F[i+c][j+c]*G[i+c][j+c]);
                }
            }
        }
#endif
        dealloc_array_fl_complex(F,m+w-1);
        dealloc_array_fl_complex(G,m+w-1);
        dealloc_array_fl_complex(H,m+w-1);
    }

    dealloc_array_fl_complex(F1,m+w-1);
    free(coff);
    /* Compute Output Image from P and Q */
    for (i=0;i<m;i++)
    {
        for (j=0;j<n;j++)
        {
            if (fabs(creal(Q[i][j]))<=0.001)
                outimg[i][j]=img[i][j];
            else 
                outimg[i][j]=(P[i][j]/Q[i][j]);
        }
    }
    
    dealloc_array_fl(P,m);
    dealloc_array_fl(Q,m);
    return EXIT_SUCCESS; 
}
