# Simulated Exposure Fusion (SEF)

Octave/Matlab implementation of _Simulated Exposure Fusion_, a single-image contrast enhancement operator.
Charles Hessel <charles.hessel@cmla.ens-cachan.fr> CMLA, ENS Paris-Saclay

This implementation is part of the IPOL publication:
> _Simulated Exposure Fusion_, Charles Hessel, In Image Processing On Line 9 (2019). https://www.ipol.im/pub/pre/279/

This method was first described in the following paper:
> HESSEL, Charles, MOREL, Jean-Michel, An Extended Exposure Fusion and its Application to Single Image Contrast Enhancement. In: 2020 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2020.

Version 1.0 released on December, 2019
Future version of this code: https://github.com/chlsl/simulated-exposure-fusion-ipol/


## Organisation

```bash
├── README.md                               # This README
├── dispersionMap.m                         # For SEF demo
├── exposureFusion                          # T. Mertens implementation of EF
│   ├── LICENSE                         (*)
│   ├── README.md                       (*)
│   ├── downsample.m                    (*)
│   ├── gaussian_pyramid.m              (*)
│   ├── laplacian_pyramid.m             (*)
│   ├── pyramid_filter.m                (*)
│   ├── reconstruct_laplacian_pyramid.m (*)
│   └── upsample.m                      (*)
├── multiscaleBlending.m                    # For SEF
├── remapFun.m                              # For SEF
├── robustNormalization.m               (*) # For SEF
├── runsef.m                                # Interface for SEF demo
└── sef.m                                   # Main file for SEF
```
`(*)`: not reviewed

**Non-reviewed code**:
- The code in the directory `exposureFusion` is written by Tom Mertens and can be found at https://github.com/Mericam/exposure-fusion (commit `03e2469`).
- The scripts `robustNormalization.m` is part of a previous IPOL publiation and have been reviewed in this context. Refer to
  > Charles Hessel, An Implementation of the Exposure Fusion Algorithm, Image Processing On Line, 8 (2018), pp. 369–387. https://doi.org/10.5201/ipol.2018.230


## Prerequisites

To run this programm, you can use either [GNU Octave](https://www.gnu.org/software/octave/) (version 4.0 or higher) or Matlab (version R2016b or higher).

For Octave, the image package should be first installed. Simply type `pkg install image` in the Octave prompt (or use `eval` as in the example below).
You will also need `gnuplot` and `fig2dev` to print the figure with Octave.
```bash
octave --eval 'pkg install -forge image'
apt-get install gnuplot fig2dev
```


## Usage

Get help by calling the program `runsef.m` without arguments. It outputs:
```
Usage: octave -W -qf runsef.m input.png output.png alpha beta nScales bClip wClip
- input  : input image name
- output : output image name
- alpha  : maximal contrast factor (recommended: 6)
- beta   : reduced dynamic range (recommended: 0.5)
- nScales: number of scales: n, or 0 or -1 or -2 for auto setting (recommended: 0)
           -  n: use n scales
           -  0: use Mertens et al. pyramid depth
           - -1: use a deeper pyramid (smallest dim has size 1 at last scale)
           - -2: use the deepest pyramid (largest dim has size 1 at last scale)
- bClip  : maximal percentage of white-saturated pixels (recommended: 1)
- wClip  : maximal percentage of black-saturated pixels (recommended: 1)
```


## Example

The command
```bash
octave -W -qf runsef.m test/input.jpg output.png 6 0.5 0 1 1
```
applies the simulated exposure fusion to `input.jpg` and save the result as `output.png`. The parameters are `alpha=6`, `beta=0.5`, `nScales=0` (i.e.  automatically computed) and 1% of clipping is authorized in the bright and dark parts of the fused image.

Some other images are written: `jointHisto.png` contains a joint histogram of the output vs. input intensities; `remapFun.png` is a plot of the different remapping functions used to simulate images from the input; `simulated<k>.png` and `weightmap<k>.png`, where `k` ranges from 1 to the number of fused images, contain the simulated images and the associated weightmaps.

With Matlab, use
```bash
matlab -nojvm -nodisplay -nosplash -batch "runsef test/input.jpg output.png 6 0.5 0 1 1"
```
from the command line, or call the function `runsef` from the Matlab prompt, with all parameters as strings.

Note: The option "-batch" has been recently introduced in Matlab. If it is not available, use
```bash
matlab -nodesktop -nodisplay -nosplash -r "try, runsef('test/input.jpg', 'output.png', '6', '0.5', '-2', '1', '1'), catch ME, fprintf('Error: %s: %s\n',ME.identifier,ME.message), end, quit"
```


## Testing

Using the above command, the result `output.png` should be identical to the provided file `test/output_expected.png`.

_Tested with the following configurations: mac os 10.12, Octave 5.1.0; mac os 10.12, Matlab R2018b; Ubuntu 18.04, Octave 4.2.2._


## Known issues

For Ubuntu users, if `sudo apt install fig2dev` complains about not being able to locate `fig2dev`, try installing `transfig` instead.

