function [D] = reglaAprendizaje (W, b, x, e, fAprendizaje)

%Recalcula la matriz de pesos.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

n = (fAprendizaje*e*x)';

D = W + n;


