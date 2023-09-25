function [hs, h, dhs, dh, Ns, N] = remapFun(alpha, beta, lambda, cval, M)
% remapFunc creates functions to remap the intensity of the input image so as to
% generate a sequence to fuse with exposure fusion.
% [hs, h, dhs, dh, Ns, N] = remapFun(alpha, beta, lambda, cval, M)
%
% Charles Hessel, CMLA, ENS Paris-Saclay.

if lambda < 0 || lambda >= 1
    error(['Incorrect value for parameter lambda. ' ...
           'Correct range: 0 <= lambda < 1']);
end

if beta <= 0 || beta > 1
    error(['Incorrect value for beta. ' ...
           'Correct range: 0 < beta <= 1.']);
end

if exist('M','var') && M ~= 0
    %%% Compute N, Ns and Nx in function of cval
    Mp = M - 1;
    Ns = floor(Mp*cval);
    N  = Mp - Ns;
    Nx = max(N,Ns);
else
    %%% Compute optimal number of images (smallest N that ensure every part of
    %%% the input range is enhanced)
    Mp = 1;              % Mp = M-1; M is the total number of images
    Ns = floor(Mp*cval); % number of images generated with fs
    N  = Mp-Ns;          % number of images generated with f
    Nx = max(N,Ns);      % used to compute maximal factor
    tmax1  = (+1    + (Ns+1)*(beta-1)/Mp)/(alpha^(1/Nx));     % t_max k=+1
    tmin1s = (-beta + (Ns-1)*(beta-1)/Mp)/(alpha^(1/Nx)) + 1; % t_min k=-1
    tmax0  = 1      + Ns*(beta-1)/Mp;                         % t_max k=0
    tmin0  = 1-beta + Ns*(beta-1)/Mp;                         % t_min k=0
    while tmax1 < tmin0 || tmax0 < tmin1s
        %%% add an image to the sequence
        Mp = Mp+1;
        %%% update values
        Ns = floor(Mp*cval);
        N  = Mp-Ns;
        Nx = max(N,Ns);
        tmax1  = (+1    + (Ns+1)*(beta-1)/Mp)/(alpha^(1/Nx));     % t_max k=+1
        tmin1s = (-beta + (Ns-1)*(beta-1)/Mp)/(alpha^(1/Nx)) + 1; % t_min k=-1
        tmax0  = 1      + Ns*(beta-1)/Mp;                         % t_max k=0
        tmin0  = 1-beta + Ns*(beta-1)/Mp;                         % t_min k=0
        if Mp > 49 % break if no solution
            warning(['The estimation of the number of image required in '...
                     'the sequence stopped because it reached M>50. ' ...
                     'Check the parameters.']);
            break
        end
    end
end
fprintf('M = %d, with N = %d and Ns = %d.\n', Mp+1, N, Ns);

%%% Remapping functions
f  = @(t,k) alpha^(+k/Nx) .* t        ; % enhance dark parts
fs = @(t,k) alpha^(-k/Nx) .* (t-1) + 1; % enhance bright parts

%%% Offset for the dynamic range reduction (function "g")
r = @(k) (1-beta/2) - (k+Ns)*(1-beta)/Mp;

%%% Reduce dynamic (using offset function "r")
a  = beta/2 + lambda;
b  = beta/2 - lambda;
g  = @(t,k) (abs(t-r(k)) <= beta/2) .* t ...
          + (abs(t-r(k)) >  beta/2) .* ...
            ( sign(t-r(k)) .*( a - lambda^2 ./ ...
              (abs(t-r(k)) - b + (abs(t-r(k))==b)) ) + r(k) );

%%% final remapping functions: h = g o f
h  = @(t,k) g(f (t, k), k); % create brighter images (k>=0) (enhance dark parts)
hs = @(t,k) g(fs(t, k), k); % create darker images (k<0) (enhance bright parts)

%%% derivative of g with respect to t
dg  = @(t,k) (abs(t-r(k)) <= beta/2) .* 1 ...
           + (abs(t-r(k)) >  beta/2) .* ...
             (lambda^2 ./ (abs(t-r(k)) - b + (abs(t-r(k))==b)).^2);

%%% derivative of the remapping functions: dh = f' x g' o f
dh  = @(t,k) alpha^(+k/Nx) .* dg(f (t, k), k);
dhs = @(t,k) alpha^(-k/Nx) .* dg(fs(t, k), k);

%%% Warning:
%%% functions h and hs can give values outside the range [0,1].
%%% The remapped images should be clipped in [0,1] after application of h or hs.
%%% Remember to set to zero the contrast weights where the images are clipped.
