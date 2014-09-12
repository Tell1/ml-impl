function [red] = configuracion(entradas, objetivos, parametros)

% Funci�n que inicializa la estructura de la red
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

W = zeros(parametros.m, size(entradas, 2));

red = struct('W', W, 'parametros', parametros);

end