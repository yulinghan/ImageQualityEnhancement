#! /bin/sh

# Set parameters
input="Tiya.jpg"
sigmas=5
sigmar=40
output="Tiya_out.jpg"
eps=0.001

# Compile FBF Project
make 

# Run executable FBF
./FBF $input $sigmas $sigmar $output $eps 

## Run executable FBF with Parameter to control stretching of the difference images 'sigmaref " as input
## Default value for sigmaref is 32
#sigmaref=32 
#./FBF $input $sigmas $sigmar $output $eps $sigmaref

## Run executable FBF to denoise image with gaussian noise of standard deviation sigman
#sigmaref=32
#noise_indicator=1
#sigman=10 
#./FBF $input $sigmas $sigmar $output $eps $sigmaref $noise_indicator $sigman

rm -rf FBF
