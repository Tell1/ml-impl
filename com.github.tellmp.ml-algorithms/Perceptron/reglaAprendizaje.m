function [D, threshold] = reglaAprendizaje (W, b, x, e, fAprendizaje)

%Recalcula la matriz de pesos y el array de bias.
%Para el recálculo del bias usamos una extensión de la regla de
%aprendizaje, en la que el nuevo bias será el anterior más el error.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

n = (fAprendizaje*x*e)';

D = W + n;

threshold = b + e;

end