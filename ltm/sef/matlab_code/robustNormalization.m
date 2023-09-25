function [v, vmin, vmax] = robustNormalization(u, wSat, bSat, dyn)
% The robustNormalization normalize the image with some saturation on both
% sides of the histogram.
%
% [v, vmin, vmax] = robustNormalization(u, wSat, bSat)
% [v, vmin, vmax] = robustNormalization(u, wSat, bSat, dyn)
%
% with : - u    : input image.
%        - wSat : percentage of saturation for the white
%        - bSat : percentage of saturation for the black
%        - dyn  : optional : max value in the output image (default: 1)
%        - v    : output image, in [0, dyn];
%        - vmin : minimal value of v before it is set to 0
%        - vmax : maximal value of v before it is set to dyn
%
% For color images, a pixel is counted as saturated as soon as at least one
% of the color channel is saturated.
% That is, contrarily to the previous implementation "ipolSCB", a color
% pixel with two saturated channels is NOT equivalent to two pixels with
% only one saturated channel.
%
% REFERENCE:
%   Simplest Color Balance, Nicolas Limare, Jose-Luis Lisani, Jean-Michel
%   Morel, Ana Belen Petro, and Catalina Sbert. Image Processing On Line, 1
%   (2011). http://dx.doi.org/10.5201/ipol.2011.llmps-scb
%
% Charles Hessel, CMLA, ENS Paris-Saclay -- November 2018.

if nargin == 3, dyn = 1; end

[H,W,D] = size(u);
N       = H*W;

if D > 1                                    % more than one channel
    u_maxChan     = max(u,[],3);            % max channel
    u_minChan     = min(u,[],3);            % min channel
    u_maxChanSort = sort(u_maxChan(:));
    u_minChanSort = sort(u_minChan(:));
    vmax          = u_maxChanSort(ceil(N-wSat*N/100));
    vmin          = u_minChanSort(floor(1+bSat*N/100));
else
    u_sort        = sort(u(:));
    vmax          = u_sort(round(N-wSat*N/100));
    vmin          = u_sort(round(1+bSat*N/100));
end

if vmax <= vmin         % in case vmax < vmin, do not invert the contrast
    v = vmax*ones(H,W); % replace by constant image
else
    v = (u-vmin).*(dyn/(vmax-vmin));
    v(u > vmax)  = dyn; % white saturation
    v(u < vmin)  = 0;   % black saturation
end

if D > 1
    wSat_count = sum(u_maxChan(:)>vmax)*100/N;
    bSat_count = sum(u_minChan(:)<vmin)*100/N;
else
    wSat_count = sum(u(:)>vmax)*100/N;
    bSat_count = sum(u(:)<vmin)*100/N;
end

fprintf('Robust Normalization:\n');
if sign(vmin) == -1                 % don't display (u - -2.78)*1.53
    ds = '+';                       % but (u + 2.78)*1.53 instead
else
    ds = '-';
end
if dyn ~= 1                         % don't display dyn if dyn == 1
    fprintf('- out = %.3f * (in %c %.3f) * %.3f\n', ...
        dyn, ds, abs(vmin), 1/(vmax-vmin));
else
    fprintf('- out = (in %c %.3f) * %.3f\n', ...
        ds, abs(vmin), 1/(vmax-vmin));
end
fprintf('- %2.3f %% of image clipped at %.3f,\n',wSat_count,dyn);
fprintf('- %2.3f %% of image clipped at %.3f.\n',bSat_count,0);
end
