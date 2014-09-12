function [D, threshold, W_ant, b_ant] = reglaAprendizaje (W, b, x, e, fAprendizaje, W_anterior, b_anterior, fMomento)

%Recalcula la matriz de pesos y el array de bias.
%Para el rec�lculo del bias usamos una extensi�n de la regla de
%aprendizaje, en la que el nuevo bias ser� el anterior m�s el error.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

W_ant = W;
b_ant = b;

%c�lculo del momento
momento_pesos = fMomento*(W-W_anterior);
momento_bias = fMomento*(b - b_anterior);

n = (fAprendizaje*e*x)';
n_bias = (fAprendizaje*e)';

D = W + n+ momento_pesos;

threshold = b + n_bias+ momento_bias;
