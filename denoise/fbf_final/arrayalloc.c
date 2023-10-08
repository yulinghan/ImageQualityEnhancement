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
 * @file arrayalloc.c
 * @brief Memory allocating and deallocating routines for 2D arrays of datatype double and double complex
 *
 * @author PRAVIN NAIR  <sreehari1390@gmail.com>
 **/

#include "headersreq.h"
double **alloc_array(int rows, int columns);
void dealloc_array_fl(double **arr,int m);
double complex **alloc_array_complex(int rows, int columns);
void dealloc_array_fl_complex(double complex **arr,int m);

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
double **alloc_array(int rows, int columns)
{
    int i;
    int j;
    /* Allocate an array of pointers with size equal to number of rows */
    double **twoDary = (double **) (calloc(rows,sizeof(double *)));
    double *currentrow;
 
    /* For each row, allocate an array with size equal to number of columns */
    for ( i = 0; i < rows; i++ ){
        *(twoDary + i) =  (calloc(columns,sizeof(double)));
    }

    /* Initialize the 2D array with zeros */
    for (j = 0; j < rows; j++) {
        currentrow = *(twoDary + j);
        for ( i = 0; i < columns; i++ ) {
            *(currentrow + i) = 0.0;
        }
    }
    return twoDary;
}


/**
 * \brief Deallocate dynamically allocated 2D array of doubles
 * \param arr       Pointer to 2D array
 * \param m         Number of rows
 *
 * This routine deallocates heap memory allocated for
 * 2D array of rows m and datatype double.
 */
void dealloc_array_fl(double **arr,int m)
{
    int k;
    /* Free memory corresponding to each row */
    for(k=0;k<m;k++)
    {
        free(arr[k]);
    }
    /* Free memory corresponding to the array of pointers to rows */
    free(arr);
}

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
double complex **alloc_array_complex(int rows, int columns)
{
    int i;
    int j;
    /* Allocate an array of pointers with size equal to number of rows */
    double complex **twoDary = (double complex**) (calloc(rows,sizeof(double complex *)));
    double complex *currentrow;

    /* For each row, allocate an array with size equal to number of columns */
    for ( i = 0; i < rows; i++ ){
        *(twoDary + i) =  (calloc(columns,sizeof(double complex)));
    }

    /* Initialize the 2D array with zeros */
    for (j = 0; j < rows; j++) {
        currentrow = *(twoDary + j);
        for ( i = 0; i < columns; i++ ) {
            *(currentrow + i) = 0.0+0.0*I;
        }
    }
    return twoDary;
}
/**
 * \brief Deallocate dynamically allocated 2D array of complex doubles
 * \param arr       Pointer to 2D array
 * \param m         Number of rows
 *
 * This routine deallocates heap memory allocated for
 * 2D array of rows m and datatype double complex.
 */
void dealloc_array_fl_complex(double complex **arr,int m)
{
    int k;
    /* Free memory corresponding to each row */
    for(k=0;k<m;k++)
    {
        free(arr[k]);
    }
    /* Free memory corresponding to the array of pointers to rows */
    free(arr);
}

