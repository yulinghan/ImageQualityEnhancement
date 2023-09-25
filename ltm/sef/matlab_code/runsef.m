function runsef(varargin)
% Apply SEF (Simulated Exposure Fusion) and write result.
% This function can be run with GNU Octave and Matlab.
%
% function runsef(inputName, outputName, alpha, nScales, beta, bClip, wClip)
%
% To get help, call the function without parameters.
% Several supplementary output images are dumped:
% - all the simulated images;
% - all their weightmaps;
% - a plot of the remapping functions;
% - the joint histogram between the input and output images.
%
% Charles Hessel, CMLA, ENS Paris-Saclay.
% Associated to an IPOL paper (https://www.ipol.im/pub/pre/279/), see README.

if exist('OCTAVE_VERSION', 'builtin')       % we're in Octave
    pkg load image
    arg_list = argv();
    warning('off', 'Octave:legacy-function');
else                                        % we're in Matlab
    arg_list = varargin;
    viridis = @parula;                      % replace Octave viridis colormap
end

%%% Add files to path (so that script can be called from outside its directory)
[scriptPath, scriptName, scriptExt] = fileparts(mfilename('fullpath'));
addpath( scriptPath, [scriptPath '/exposureFusion'] )

%%% Input parameters
usage = sprintf([...
  'octave -W -qf runsef.m input.png output.png alpha beta nScales bClip wClip\n' ...
  '- input  : input image name\n' ...
  '- output : output image name\n' ...
  '- alpha  : maximal contrast factor (recommended: 6)\n' ...
  '- beta   : reduced dynamic range (recommended: 0.5)\n' ...
  '- nScales: number of scales: n, or 0 or -1 or -2 for auto setting (recommended: 0)\n' ...
  '           -  n: use n scales\n' ...
  '           -  0: use Mertens et al. pyramid depth\n' ...
  '           - -1: use a deeper pyramid (smallest dim has size 1 at last scale)\n' ...
  '           - -2: use the deepest pyramid (largest dim has size 1 at last scale)\n' ...
  '- bClip  : maximal percentage of white-saturated pixels (recommended: 1)\n' ...
  '- wClip  : maximal percentage of black-saturated pixels (recommended: 1)\n']);

if isempty(arg_list), fprintf(usage); quit; end
if length(arg_list) < 6, error('Missing argument(s).\nUsage:\n%s\n', usage);
else
    alpha   = str2double(arg_list{3});      % maximal contrast factor
    beta    = str2double(arg_list{4});      % reduced dynamic range
    nScales = str2double(arg_list{5});      % number of scales in the pyramids
    bClip   = str2double(arg_list{6});      % robustNorm: black clipping
    wClip   = str2double(arg_list{7});      % robustNorm: white clipping
end

%%% Read input image
img = im2double(imread(arg_list{1}));

%%% Apply Simulated Exposure Fusion
tic
[out, int, fun] = sef(img, alpha, beta, nScales);
fprintf('Timing: Simulated Exposure Fusion: %.2f seconds\n', toc);

%%% Normalize result
tic
nor = robustNormalization(out, wClip, bClip);
fprintf('Timing: Robust Normalization: %.2f seconds\n', toc);

%%% Write output
tic
imwrite(uint8(255*nor), arg_list{2});
fprintf('Timing: Writing output: %.2f seconds\n', toc);

%%% Print Generated images and associated weights
tic
M = fun.N + fun.Ns + 1;
for n = 1:M
    imwrite(uint8(255*int.uh(:,:,n)),sprintf('simulated%d.png',n-1));
    imwrite(uint8(255*int.wh(:,:,n)),sprintf('weightmap%d.png',n-1));
end
fprintf('Timing: Writing simulated images and weights: %.2f seconds\n', toc);

%%% Compute "dispersion map"
tic
dMap = dispersionMap(max(0,min(1,imresize(sum(img,3)/3,0.5))),...
                     max(0,min(1,imresize(sum(nor,3)/3,0.5))));

%%% Write correspondance histogram (dispersion map)
imwrite(uint8(255*dMap/max(dMap(:))), viridis(256), 'jointHisto.png');
fprintf('Timing: Joint Histogram: %.2f seconds\n', toc);

%%% Prepare remapping functions
tic
x = (0:511)/511;
remap = NaN(512, M);
legCont = cell(1, M);
for k = -fun.Ns:fun.N
    n = k + fun.Ns + 1;
    if k < 0, remap(:,n) = fun.hs(x, k);    % Apply remapping function
    else,     remap(:,n) = fun.h(x, k);     % Apply remapping function
    end
    clipsup = remap(:,n) > 1;               % Detect values outside [0,1]
    clipinf = remap(:,n) < 0;
    remap(clipsup,n) = 1;                   % Clip them
    remap(clipinf,n) = 0;
    legCont{n} = sprintf('k=%d',k);         % Legend for plot
end

%%% Print remaping functions
colororder = repmat([...                    % Octave
    0         0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840],[9 1]);
fh = figure('visible','off');
for n=1:M, plot(x,remap(:,n),'Color',colororder(n,:),'LineWidth',2); hold on;
end; hold off; axis([0 1 0 1]); axis square
lh = legend(legCont,'Location','SouthEast'); set(lh,'FontSize',10);
set(gca,'position',[0 0 1 1],'units','normalized')
set(gcf,'PaperUnits','Inches','PaperPosition',[0 0 5.12 5.12])
print('-dpng','remapFun.png','-r100');
fprintf('Timing: Print remapping functions: %.2f seconds\n',toc);

%%% Give ipol the number of generated images
fileID = fopen('algo_info.txt','w');
fprintf(fileID,'nb_outputs=%d',M);
fclose(fileID);
