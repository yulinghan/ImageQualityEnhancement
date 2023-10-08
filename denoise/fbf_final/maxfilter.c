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
 * @file maxfilter.c
 * @brief Algorithm to calculate local maximum at every pixel of image
 *
 * @author PRAVIN NAIR  <sreehari1390@gmail.com>
 **/

#include"headersreq.h"

/**
 * \brief Find T = max{x} ( max{||y||<=R} (|f(x-y)-f(x)|) )
 *        (using Max-Filter Algorithm)
 * \param fin       Pointer to input image
 * \param w         Wdith of spatial kernel
 * \param m         Image height
 * \param n         Image width
 * \return T
 *
 * This routine computes the maximum value T input to the
 * range filter for the input image fin and spatial kernel
 * of width w.
 * Max-Filter Algorithm is used to calculate local maximums.  
 */
double maxfilterfind(double **fin,int w,int m,int n)
{
    /** \brief Radius of spatial kernel */
    int c=(w-1)/2;  
    double T=0.0,temp;
    
    /* Pad to divide rows and columns into integral number of max-filter windows */
    int rowceilvalue=(int)ceil((double)m/w);
        int columnceilvalue=(int)ceil((double)n/w);
    int rowpad=(rowceilvalue*w)-m;
    int columnpad=(columnceilvalue*w)-n;
    int mpad=m+rowpad;
    int npad=n+columnpad;
        double **template=alloc_array(mpad,npad);
    int i,j,k;
    double r,l;
    for (i=0;i<m;i++)
    {
        for (j=0;j<n;j++)
        {
            template[i][j]=fin[i][j];
        }
    }
    
    double *L,*R;
    
    /* Max-Filter Algorithm is applied along rows */
    for (i=0;i<m;i++)
    {
        /** \brief Arrays to store local running maximums from left and right */
        L=calloc(npad,sizeof(double));
        R=calloc(npad,sizeof(double));
        
        for (k=0;k<npad;k++)
        {
            if ((k%w)==0)
            {
                /* Reset the recursion at boundary of parition */
                L[k]=template[i][k];
                R[npad-1-k]=template[i][npad-1-k];
            }
            else
            {   
                /* Running maximum */
                L[k] = max(L[k-1],template[i][k] );
                R[npad-1-k]  = max(R[npad-k], template[i][npad-1-k] );              
            }
        }
        
        /* Compute local maximums along rows from the 2 local running maximums */
        for (k=0;k<npad;k++)
        {
            if (k-c<0)
                r=0;
            else
                r=R[k-c];
            if (k+c>(npad-1))
                l =0;
            else
                l = L[k+c];
            /* Store in template */
            template[i][k]=max(r,l);
        }   
        free(L);
        free(R);            
    }
    
    /* Max-Filter Algorithm is applied along columns */
    for (j=0;j<n;j++)
    {
        /** \brief Arrays to store local running maximums from top and bottom */
        L=calloc(mpad,sizeof(double));
        R=calloc(mpad,sizeof(double));
                
                
        for (k=0;k<mpad;k++)
        {
            if ((k%w)==0)
            {
                /* Reset the recursion at boundary of parition */
                L[k]=template[k][j];
                R[mpad-1-k]=template[mpad-1-k][j];
            }
            else
            {
                /* Running maximum */
                L[k] = max(L[k-1],template[k][j] );
                R[mpad-1-k]  = max(R[mpad-k], template[mpad-1-k][j] );
            }
        }
               
        /* Compute local maximums along columns from the 2 local running maximums */
        /* This gives the local maximums over 2D spatial kernels, from which T is computed */
        for (k=0;k<mpad;k++)
        {
            if (k-c<0)
                r=0;
            else
                r=R[k-c];
            if (k+c>(npad-1))
                l =0;
            else
                l = L[k+c];
            if (k<m)
                temp = max(r,l) - fin[k][j];
            if (temp > T)
                T = temp;
        }
        free(L);
        free(R);
    }
    return T;
}

