% BM3D image denoising.

# ABOUT

* Author    : Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
* Copyright : (C) 2011 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : GPL v3+, see GPLv3.txt

# OVERVIEW

This source code provides an implementation of the BM3D image denoising.

# UNIX/LINUX/MAC USER GUIDE

The code is compilable on Unix/Linux and Mac OS. 

- Compilation. 
Automated compilation requires the cmake program.

- Libraries. 
This code requires the libpng libtiff libjpeg and libfftw librarers.

- Image formats. 
PNG, JPEG, TIFF (including floating point) formatis is supported. 
 
-------------------------------------------------------------------------
Usage:
1. Download the code package and extract it. Go to that directory. 

2. Compile the source code (on Unix/Linux/Mac OS). 
mkdir build; cd build; cmake ..; make

3. Run BM3D image denoising.
./bm3d
The generic way to run the code is:

./bm3d input.png sigma ImDenoised.png [ImBasic.png ]
-tau_2d_hard 2DtransformStep1 -useSD_hard 
-tau_2d_wien 2DtransformStep2 -useSD_wien 
-color_space ColorSpace

with :
- cinput.png is a noisy image;
- sigma is the value of the noise;
- ImBasic.png will contain the result of the first step of the algorithm;
- ImDenoised.png will contain the final result of the algorithm;
- 2DtransformStep1: choice of the 2D transform which will be applied in the 
     second step of the algorithm. You can choose the DCT transform or the 
     Bior1.5 transform for the 2D transform in the step 1 (tau_2D_hard = dct or bior) 
     and/or the step 2. (tau_2d_wien = dct or bior).
- useSD_hard: for the first step, users can choose if they prefer to use
     standard variation for the weighted aggregation (useSD1 = 1)
- 2DtransformStep2: choice of the 2D transform which will be applied in the 
     second step of the algorithm. You can choose the DCT transform or the 
     Bior1.5 transform for the 2D transform in the step 1 (tau_2D_hard = dct or bior) 
     and/or the step 2. (tau_2d_wien = dct or bior).
- useSD_wien: for the second step, users can choose if they prefer to use
     standard variation for the weighted aggregation (useSD2 = 1)
- ColorSpace: choice of the color space on which the image will be applied. 
     you can choose the colorspace for both steps between : rgb, yuv, ycbcr and opp.
- patch_size: overrides the default patch size
- nb_threads: specifies the number of working threads
- verbose: print additional information

Example, run
./BM3Ddenoising cinput.png 10 ImDenoised.png ImBasic.png -useSD_wien \
   -tau_2d_hard bior -tau_2d_wien dct -color_space opp

# ABOUT THIS FILE

Copyright 2011 IPOL Image Processing On Line http://www.ipol.im/

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.
