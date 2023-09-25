function [ v, int, fun ] = sef(u, alpha, beta, nScales, M, med, lambda)
% sef applies the Simulated Exposure Fusion.
%
% Usage:
%   v = sef(u)
% Or
%   [v, int, fun] = sef(u, alpha, beta, nScales, M, med, lambda)
%
% Only the first parameter is mandatory.
% Replace the optional values by []: the order matters.
% Example: v = sef(u, 6, 0.5, [], [], 0) set alpha, beta and med. The other
% empty values are set to the default.
%
% Inputs:
%   - u      : input image, double, in [0, 1]
%   - alpha  : (default: 6) maximal contrast factor ( alpha >= 1 )
%   - beta   : (default: 0.5) reduced dynamic range ( 0 < beta <= 1 )
%   - nScales: (default: 0) number of scales used in the fusion.
%              0,-1, and -2 mean auto. 0 will compute the depth in the same way
%              as Mertens et al. implementation. The residual is often a few
%              pixels wide in both dimensions. (-1) will make the smallest
%              dimension be of size 1 in the residual; with (-2) this will be
%              the largest (and thus the smallest too).
%
% Inputs (advanced):
%   - M      : (default: 0, i.e. auto) use this to force a specific number of
%              images in the sequence.
%   - med    : (default: NaN, i.e. auto) use this to force the proportion of
%              dark and bright images
%   - lambda : (default: 0.125) a constant used to control the remapping
%              functions' shape (controls the speed of the decay outside the
%              restrained range). Not recommended to change it.
%
% Outputs:
%   - v  : output, enhanced image
%   - int: struct:
%          - int.uh: generated input sequence (\hat{u})
%          - int.wh: associated weights (\hat{w})
%   - fun: struct containing all used functions and N* and N:
%          - fun.hs  = g o f*          for under-exposed images
%          - fun.h   = g o f           for over-exposed images
%          - fun.dhs = d( g o f* )/dt  contrast metric
%          - fun.dh  = d( g o f )/dt   contrast metric
%          - Ns = N*                   scalar
%          - N                         scalar
%
% Charles Hessel, CMLA, ENS Paris-Saclay.
% December 2019


%%% options handling

% number of inputs and outputs
narginchk(1,7)
nargoutchk(1,3)

% defaults parameters
if ~exist('alpha','var')   || isempty(alpha),   alpha = 6;     end
if ~exist('beta','var')    || isempty(beta),    beta = .5;     end
if ~exist('nScales','var') || isempty(nScales), nScales = 0;   end
if ~exist('M','var')       || isempty(M),       M = 0;         end
if ~exist('med','var')     || isempty(med),     med = NaN;     end
if ~exist('lambda','var')  || isempty(lambda),  lambda = .125; end

% check bounds
if alpha < 1,             error('sef requires alpha >= 1');    end
if beta <= 0 || beta > 1, error('sef requires 0 < beta <= 1'); end
if med < 0 || med > 1,    error('sef requires 0 <= med <= 1'); end


%%% Convert to hsv colorspace

if size(u, 3) == 3                      % color input image
  z = rgb2hsv(u);
  l = z(:,:,3);                         % luminance ("value" channel of hsv)
else                                    % gray input image
  l = u;
end
c = u ./ (l + 2^(-16));                 % Because of Octave: color coefficients


%%% Compute remapping functions

if isnan(med), med = median(u(:)); end  % auto med

[hs, h, dhs, dh, Ns, N] = remapFun( alpha, beta, lambda, med, M );


%%% Simulate a sequence from image u. Compute the contrast weights

seq = NaN([size(l) N+Ns+1]);
wc  = NaN([size(l) N+Ns+1]);
for k = -Ns:N
    if k < 0
        seq(:,:,k+Ns+1) = hs( l, k );   % Apply remapping function
        wc(:,:,k+Ns+1) = dhs( l, k );   % Compute contrast measure
    else
        seq(:,:,k+Ns+1) = h( l, k );    % Apply remapping function
        wc(:,:,k+Ns+1) = dh( l, k );    % Compute contrast measure
    end
end

clipsup = seq > 1;                      % Detect values outside [0,1]
clipinf = seq < 0;
seq(clipsup) = 1;                       % Clip them
seq(clipinf) = 0;
wc(clipsup) = 0;                        % Set to 0 contrast of clipped values
wc(clipinf) = 0;


%%% Well-exposedness weights

we = zeros(size(seq));
for n = 1:(N+Ns+1)
    we(:,:,n) = exp( -.5 * (seq(:,:,n) - .5).^2 / .2^2 );
end


%%% Final normalized weights

w = wc .* we + eps;
w = w ./ sum(w,3);


%%% multiscale blending

lp = multiscaleBlending(seq, w, nScales);


%%% Restore color

% v = hsv2rgb(cat(3,z(:,:,1:2),lp));    % Octave implementation of hsv2rgb
v = lp .* c;                            % doesn't take values outside [0,1].
                                        % Using color coeffs is equivalent.


%%% Other outputs

int.uh = seq;   % \hat{u}
int.wh = w;     % \hat{w}
int.l  = l;     % \ell
int.lp = lp;    % \ell'
fun.h = h;      % g o f
fun.hs = hs;    % g o f*
fun.dh = dh;    % (g o f)'
fun.dhs = dhs;  % (g o f*)'
fun.N = N;      % N
fun.Ns = Ns;    % N*
