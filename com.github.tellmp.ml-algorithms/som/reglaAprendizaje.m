function D = reglaAprendizaje(x, e, fAprendizaje, v, W)

% Funci�n que adapta los pesos.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

 D = W + fAprendizaje * v * e;