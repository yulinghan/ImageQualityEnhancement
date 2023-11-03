#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include <vector>
#include <fftw3.h>

//! Add boundaries by symetry
void symetrize(
    const std::vector<float> &img
,   std::vector<float> &img_sym
,   const unsigned width
,   const unsigned height
,   const unsigned N
);

//! Look for the closest power of 2 number
int closest_power_of_2(
    const unsigned n
);

//! Initialize a set of indices
void ind_initialize(
    std::vector<unsigned> &ind_set
,   const unsigned max_size
,   const unsigned N
,   const unsigned step
);

//! For convenience
unsigned ind_size(
    const unsigned max_size
,   const unsigned N
,   const unsigned step
);

//! Initialize a 2D fftwf_plan with some parameters
void allocate_plan_2d(
    fftwf_plan* plan
,   const unsigned N
,   const fftwf_r2r_kind kind
,   const unsigned nb
);
#endif // UTILITIES_H_INCLUDED
