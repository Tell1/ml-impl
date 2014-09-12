function D = reglaAprendizaje(x, e, fAprendizaje, v, W)

% Función que adapta los pesos.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

 D = W + fAprendizaje * v * e;