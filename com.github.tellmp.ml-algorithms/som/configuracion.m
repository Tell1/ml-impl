function [red] = configuracion(entradas, objetivos, parametros)

% Función que inicializa la estructura de la red
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

W = zeros(parametros.m, size(entradas, 2));

red = struct('W', W, 'parametros', parametros);

end