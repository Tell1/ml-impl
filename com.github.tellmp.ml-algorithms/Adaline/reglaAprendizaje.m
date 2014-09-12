function [D] = reglaAprendizaje (W, b, x, e, fAprendizaje)

%Recalcula la matriz de pesos.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

n = (fAprendizaje*e*x)';

D = W + n;


