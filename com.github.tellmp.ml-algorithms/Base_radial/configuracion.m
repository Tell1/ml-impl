function [red] = configuracion(entradas, objetivos, parametros)

% Funci�n que inicializa la estructura de la red
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

numEntradas = size(entradas,1);
numSalidas = size(objetivos,1);

red = struct('numEntradas', numEntradas, 'numSalidas', numSalidas, 'parametros', parametros, 'W', 0, 'b', 0, 'centros', 0, 'distancia', 0);


end

