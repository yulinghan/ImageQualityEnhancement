% Separable Non-Local Means
%
%  Authors: Sanjay Ghosh and Kunal N. Chaudhury.
%
%  Date:  Oct 25, 2017.
%
%  References:
%
%   [1] S. Ghosh and K. N. Chaudhury, "Artifact reduction for separable
%       non-local means," SPIE Journal of Electronic Imaging Letters, 2017.
%
%   [2] S. Ghosh and K. N. Chaudhury, “Fast Separable Non-Local Means,”
%       SPIE Journal of Electronic Imaging, vol. 25, no. 2, 023026, 2016.
%       arXiv: https://arxiv.org/abs/1407.2343
%
%   contact: sanjayg@iisc.ac.in, kunal@iisc.ac.in
%
clc; close all; clear
% noise and NLM parameters
sigmaN = 30;
S = 10;
K = 3;
alpha = 10 * sigmaN;
% input grayscale image
x = double(imread('./images/peppers256.png'));
% x = double(imread('./images/house256.png'));
% x = double(imread('./images/man512.png'));
[m,n] = size(x);
% add noise
f = x + sigmaN*randn(m,n);
% local parameters
beta = alpha / sqrt(2*K + 1);
delta = 0.5;
beta_1 = delta * beta;
box = ones(1, 2*K+1);
S1 = 9;
S2 = 4;
% initialization
f_tilde = zeros(size(x));
f_hat = zeros(size(x));
% rowwise processing
y_pad = padarray(f, [S1,0] ,'symmetric');
for i = 1:m
    temp = y_pad(i + S1 + (-S1:S1),:);
    f_tilde(i,:) = multipatchpacking(temp, S1, S1, K, beta, box);     
end
% columnwise processing
f_tilde_pad = padarray(f_tilde, [0, S2] ,'symmetric');
for j=1:m
    temp = f_tilde_pad(:, j + S2 + (-S2 : S2))';
    f_hat(:,j) = multipatchpacking(temp, S2, S2, K, beta_1, box)' ;
end
% results
peak = 255;
PSNR_noisy = 10 * log10(m * n * peak^2 / sum(sum((f- x).^2)) );
PSNR_proposed = 10 * log10(m * n * peak^2 / sum(sum((f_hat - x).^2)) );
SSIM_noisy = 100 * ssim(f, x);
SSIM_proposed = 100 * ssim(f_hat, x);
fprintf('PSNR of the noisy image is %.2f dB. \n', PSNR_noisy);
fprintf('PSNR of the denoised image is %.2f dB. \n', PSNR_proposed);
fprintf('\n');
fprintf('SSIM of the noisy image is %.2f dB. \n', SSIM_noisy);
fprintf('SSIM of the denoised image is %.2f dB. \n', SSIM_proposed);
figure; imshow(uint8(f)); 
figure; imshow(uint8(f_hat));