function [D, threshold] = reglaAprendizaje (W, b, x, e, fAprendizaje)

%Recalcula la matriz de pesos y el array de bias.
%Para el rec�lculo del bias usamos una extensi�n de la regla de
%aprendizaje, en la que el nuevo bias ser� el anterior m�s el error.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

n = (fAprendizaje*x*e)';

D = W + n;

threshold = b + e;

end