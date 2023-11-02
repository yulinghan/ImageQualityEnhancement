/*
 * Copyright (c) 2011, Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
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
 * @file utilities.cpp
 * @brief Utilities functions
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "utilities.h"

#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3

using namespace std;

/**
 * @brief Add boundaries by symetry
 *
 * @param img : image to symetrize
 * @param img_sym : will contain img with symetrized boundaries
 * @param width, height, chnls : size of img
 * @param N : size of the boundary
 *
 * @return none.
 **/
void symetrize(
    const std::vector<float> &img
,   std::vector<float> &img_sym
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
){
    //! Declaration
    const unsigned w = width + 2 * N;
    const unsigned h = height + 2 * N;

    if (img_sym.size() != w * h * chnls)
        img_sym.resize(w * h * chnls);

    for (unsigned c = 0; c < chnls; c++)
    {
        unsigned dc = c * width * height;
        unsigned dc_2 = c * w * h + N * w + N;

        //! Center of the image
        for (unsigned i = 0; i < height; i++)
            for (unsigned j = 0; j < width; j++, dc++)
                img_sym[dc_2 + i * w + j] = img[dc];

        //! Top and bottom
        dc_2 = c * w * h;
        for (unsigned j = 0; j < w; j++, dc_2++)
            for (unsigned i = 0; i < N; i++)
            {
                img_sym[dc_2 + i * w] = img_sym[dc_2 + (2 * N - i - 1) * w];
                img_sym[dc_2 + (h - i - 1) * w] = img_sym[dc_2 + (h - 2 * N + i) * w];
            }

        //! Right and left
        dc_2 = c * w * h;
        for (unsigned i = 0; i < h; i++)
        {
            const unsigned di = dc_2 + i * w;
            for (unsigned j = 0; j < N; j++)
            {
                img_sym[di + j] = img_sym[di + 2 * N - j - 1];
                img_sym[di + w - j - 1] = img_sym[di + w - 2 * N + j];
            }
        }
    }

    return;
}

/**
 * @brief Look for the closest power of 2 number
 *
 * @param n: number
 *
 * @return the closest power of 2 lower or equal to n
 **/
int closest_power_of_2(
    const unsigned n
){
    unsigned r = 1;
    while (r * 2 <= n)
        r *= 2;

    return r;
}

/**
 * @brief Initialize a set of indices.
 *
 * @param ind_set: will contain the set of indices;
 * @param max_size: indices can't go over this size;
 * @param N : boundary;
 * @param step: step between two indices.
 *
 * @return none.
 **/
void ind_initialize(
    vector<unsigned> &ind_set
,   const unsigned max_size
,   const unsigned N
,   const unsigned step
){
    ind_set.clear();
    unsigned ind = N;
    while (ind < max_size - N)
    {
        ind_set.push_back(ind);
        ind += step;
    }
    if (ind_set.back() < max_size - N - 1)
        ind_set.push_back(max_size - N - 1);
}

/**
 * @brief For convenience. Estimate the size of the ind_set vector built
 *        with the function ind_initialize().
 *
 * @return size of ind_set vector built in ind_initialize().
 **/
unsigned ind_size(
    const unsigned max_size
,   const unsigned N
,   const unsigned step
){
    unsigned ind = N;
    unsigned k = 0;
    while (ind < max_size - N)
    {
        k++;
        ind += step;
    }
    if (ind - step < max_size - N - 1)
        k++;

    return k;
}

/**
 * @brief Initialize a 2D fftwf_plan with some parameters
 *
 * @param plan: fftwf_plan to allocate;
 * @param N: size of the patch to apply the 2D transform;
 * @param kind: forward or backward;
 * @param nb: number of 2D transform which will be processed.
 *
 * @return none.
 **/
void allocate_plan_2d(
    fftwf_plan* plan
,   const unsigned N
,   const fftwf_r2r_kind kind
,   const unsigned nb
){
    int            nb_table[2]   = {N, N};
    int            nembed[2]     = {N, N};
    fftwf_r2r_kind kind_table[2] = {kind, kind};

    float* vec = (float*) fftwf_malloc(N * N * nb * sizeof(float));
    (*plan) = fftwf_plan_many_r2r(2, nb_table, nb, vec, nembed, 1, N * N, vec,
                                  nembed, 1, N * N, kind_table, FFTW_ESTIMATE);

    fftwf_free(vec);
}
