function [red] = configuracion(entradas, objetivos, parametros)

% Función que inicializa la estructura de la red
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

numEntradas = size(entradas,1);
numSalidas = size(objetivos,1);

red = struct('numEntradas', numEntradas, 'numSalidas', numSalidas, 'parametros', parametros, 'W_1', 0, 'b_1', 0, 'W_2', 0, 'b_2', 0);


end

