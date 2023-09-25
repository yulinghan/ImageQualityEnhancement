function out = multiscaleBlending(seq, W, nScales)
% multiscaleBlending
%
% out = multiscaleBlending(seq, W, nScales)
%
% nScales can have several values
%  .   n, with n>0 : number of scales
%  .   0  auto (default). Using Mertens's et al. way to compute the depth
%  . (-1) auto nSCales, such that min(hp,wp)==1
%  . (-2) auto nSCales, such that max(hp,wp)==1 (deepest possible pyramid)
%
% Charles Hessel, CMLA, ENS Paris-Saclay
% Adapted from T. Mertens exposure_fusion.m

if ~exist('nScales','var'), nScales = 0; end

if nScales ==  0, autoRef = true; else autoRef = false; end
if nScales == -1, autoMin = true; else autoMin = false; end
if nScales == -2, autoMax = true; else autoMax = false; end

[h,w,n] = size(seq);

if autoRef || autoMin || autoMax                % automatic setting of parameter

    nScRef = floor( log(min(h,w)) / log(2) );   % Mertens's et al. value

    nScales = 1;                                % Initializations
    hp = h;
    wp = w;

    while autoRef && (nScales < nScRef) || ...  % stops at nScRef
          autoMin && (hp > 1 && wp > 1) || ...  % stops at min(hp,wp)==1
          autoMax && (hp > 1 || wp > 1)         % stops at max(hp,wp)==1
        nScales = nScales + 1;
        hp = ceil(hp/2);
        wp = ceil(wp/2);
    end
    fprintf('Number of scales: %d; residual''s size: %dx%d\n', ...
        nScales, hp, wp);
end

% allocate memory for pyr
pyr = cell(nScales,1);
hp = h;
wp = w;
for scale = 1:nScales
    pyr{scale} = zeros(hp,wp);
    hp = ceil(hp/2);
    wp = ceil(wp/2);
end

% multiresolution blending
for i = 1:n
    % construct pyramid from each input image
    pyrW = gaussian_pyramid(W(:,:,i),nScales);
    pyrI = laplacian_pyramid(seq(:,:,i),nScales);

    % blend
    for scale = 1:nScales
        pyr{scale} = pyr{scale} + pyrW{scale}.*pyrI{scale};
    end
end

% reconstruct
out = reconstruct_laplacian_pyramid(pyr);
