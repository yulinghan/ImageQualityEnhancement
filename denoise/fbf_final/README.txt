% Fast bilateral filter.

# ABOUT

* Author    : Pravin Nair <sreehari1390@gmail.com> , Anmol Popli <anmol.ap020@gmail.com>
* Copyright : (C) 2016 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : GPL v3+, see GPLv3.txt
* Version   : 1.0 2013

# OVERVIEW

This source code provides an implementation of fast version of bilateral filter.

# UNIX/LINUX

The code is compilable on Unix/Linux. 

- Compilation. 
Automated compilation requires the make program.

- Library. 
This code requires the libpng, libjpeg and libtiff library.

-API.
This code requires OpenMp application interface

- Image format. 
Formats supported are jpeg , png and tiff. 
 
-------------------------------------------------------------------------
Usage:
1. Download the code package and extract it. Go to that directory. 

2. Compile the source code (on Unix/Linuxi).
There are two ways to compile the code.
(1) RECOMMENDED, with Open Multi-Processing multithread parallelization
(http://openmp.org/). Roughly speaking, it accelerates the program using the
multiple processors in the computer. Run
make OMP=1

OR
(2) If the complier does not support OpenMp, run
make

3. Run Fast bilateral filter.

./FBF cinput.png sigmas sigmar coutput.png eps

with :
- cinput.png is a gray scaleimage;
- sigmas is standard deviation of spatial gaussian kernel;
- sigmar is standard deviation of range gaussian kernel;
- coutput.png - output image after applying bilateral filter;
- eps - lower bound on filter approximation error. 

Usage of demo file:

In demo.sh , change the parameters as:
input as "full name of input image";
sigmas as value of spatial gaussian kernel deviation;
sigmar as value of range gaussian kernel deviation;
output as "full name of output image";
eps as lower bound on filter approximation error.

Run sh demo.sh

Functionality to save difference image between original image and filtered image is added.
Additional commands(commented) has been added in demo.sh with following functionalities

1) Parameter to control stretching of the difference images 'sigmaref " can be given as input
./FBF cinput.png sigmas sigmar coutput.png eps sigmaref
with :
- sigmaref is Parameter to control stretching of the difference image (default is 32);

2) Denoising functionality is added when image is affected by gaussian noise
./FBF $input $sigmas $sigmar $output $eps $sigmaref $noise_indicator $sigman
with:
- noise_indicator is indicator function to add noise if 1 (default is 0);
- sigman is standard deviation of gaussian noise to be added.

# ABOUT THIS FILE

Copyright 2011 IPOL Image Processing On Line http://www.ipol.im/

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.

Thanks
------

Comments with suggestions, errors, bugs or strange results are welcome.
