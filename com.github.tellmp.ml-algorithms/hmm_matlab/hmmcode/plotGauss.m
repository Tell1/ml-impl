function [hh] = plotGauss(mu1,mu2,var1,var2,covar,plopt,plopt2)
%plotGauss(mu1,mu2,var1,var2,covar,  plopt,plopt2)
%
%PLOT A 2D Gaussian
%This function plots the given 2D gaussian on the current plot.

if(nargin<7) plopt2 = 'y+'; end
if(nargin<6) plopt = 'y'; end

t = -pi:.01:pi;
k = length(t);
x = sin(t);
y = cos(t);

%R = sqrt([var1 covar; covar var2]);
R = [var1 covar; covar var2];

[vv,dd] = eig(R);
A = real((vv*sqrt(dd))');
z = [x' y']*A;

holdss = ishold;
hold on;
hh = plot(z(:,1)+mu1,z(:,2)+mu2,plopt);
plot(mu1,mu2,plopt2);
if(holdss~=1) hold off; end
