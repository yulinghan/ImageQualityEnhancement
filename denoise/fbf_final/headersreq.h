#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <complex.h>
#include <tgmath.h>
#include <math.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include "mt19937ar.h"
#include <string.h>
#include "imageio.h"
#include "gnuplot_i.h"
#ifdef __linux__
#include <sched.h>
#endif
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

/** \brief struct of program parameters */
typedef struct
{
    /** \brief Number of coefficients */
    int K;
    /** \brief Array to store coefficients */
    double* coeff;
    /** \brief T computed by maxfilter */
    double T;
} program_params;
/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Main function

#ifdef __linux__
/**
 * \brief Compute the number of physical cores and assign one thread to each
 * \return number of physical cores
 *
 * This routine computes the number of physical cores on
 * the system from the system cpu info.
 * It then sets number of threads equal to the number of
 * cores and inside a parallel region assigns the threads
 * to hyperthreads on distinct physical cores.
 */
int core_affinity();
#endif

/**
 * \brief Dynamically allocate 2D array of doubles
 * \param rows      Number of rows
 * \param columns   Number of columns
 * \return pointer to 2D array
 *
 * This routine allocates memory in heap for a 2D
 * array of dimensions rows x columns and datatype
 * double.
 */
double **alloc_array(int rows, int columns);

/**
 * \brief Deallocate dynamically allocated 2D array of doubles
 * \param arr       Pointer to 2D array
 * \param m         Number of rows
 *
 * This routine deallocates heap memory allocated for
 * 2D array of rows m and datatype double.
 */
void dealloc_array_fl(double **arr,int m);

/**
 * \brief Dynamically allocate 2D array of complex doubles
 * \param rows      Number of rows
 * \param columns   Number of columns
 * \return pointer to 2D array
 *
 * This routine allocates memory in heap for a 2D
 * array of dimensions rows x columns and datatype
 * double complex.
 */
double complex **alloc_array_complex(int rows, int columns);

/**
 * \brief Deallocate dynamically allocated 2D array of complex doubles
 * \param arr       Pointer to 2D array
 * \param m         Number of rows
 *
 * This routine deallocates heap memory allocated for
 * 2D array of rows m and datatype double complex.
 */
void dealloc_array_fl_complex(double complex **arr,int m);

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
double maxfilterfind(double **fin,int w,int m,int n);

/**
 * \brief Apply 2D Gaussian filter to input image
 *        (Young and van Vliet's algorithm) 
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
 * Young and van Vliet's fast recursive algorithm.
 */
void convolve_young2D(int rows, int columns, int sigma, double complex** ip_padded);

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
void convolve_deriche2D(int rows, int columns, int sigma, double complex** ip_padded);

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
 * \return Success or Failure 
 *
 * This routine applies the fast shiftable bilateral filter
 * with parameters sigmas & sigmar to input image img of
 * dimensions m x n and computes output image outimg.
 * The algorithm used is Fourier Basis approximation.
 * The Gaussian spatial convolutions are performed using
 * Young and van Vliet's fast recursive algorithm.
 * The convolutions are performed parallelly with one thread
 * assigned for each physical core on the system.
 */
int shiftableBF(int m, int n, int sigmas, double sigmar, double** img, double** outimg, int cores, program_params* params,double eps);

/**
 * \brief Apply symmetric padding to input image
 * \param rows      Image height
 * \param columns   Image width
 * \param in Pointer to input image padded with zeros
 *
 * This routine applies mirror boundary conditions
 * to input image which is zero padded i.e size of
 * input image will be [rows+2*w, columns+2*w]
 */
void symmetric_padding(int rows,int columns,double complex **in,int w);

/**
 * \brief Calculating standard deviation of 1D array 
 * \param arr       1D array
 * \param length    length of array 'arr'
 * \return Standard deviation of array 'arr'
 *
 * This routine takes 1D input array 'arr' and
 * calculate standard deviation of the array
 */
double calculatestd(double* arr,int length);

/**
 * \brief Adding gaussian noise to input image
 * \param image     Input image
 * \param rows      Image height
 * \param columns   Image width
 * \param sigman    Standard deviation of gaussian noise to be added
 *
 * This routine adds gaussian noise of standard deviation
 * sigman to the input image
 */
int addgaussiannoise(double **image, int rows, int columns, double sigman);


