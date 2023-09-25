function dMap = dispersionMap (I, R)
% dispersionMap generates an images that shows the correspondance between
% the intensities in the input image I with those in the modified image R.
% I and R dynamic range must be in [0 1].
% dMap size is 512x512 (256x256 intensities, then upsample by factor 2)
%
% Charles Hessel, CMLA, ENS Paris-Saclay
% Created on Friday, April 6, 2018

in   = uint16(255*I(:))+1;
out  = uint16(255*R(:))+1;
dMap = zeros(256);
for n = 1:length(in)
    dMap(out(n), in(n)) = dMap(out(n), in(n)) + 1;
end
dMap = log(1+flipud(dMap));
dMap = imresize(dMap,2,'nearest');
