function error = medida_error(clases, entradas, centros)

% Funci�n quecalcula la medida del error de k-medias. Este error es la suma
% de las distancias de cada uno de los datos de entrada al centro de su
% cluster, elevado al cuadrado.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

error = 0;

for i=1:size(entradas, 1)
   clase = clases(i);
   error = error + (norm(entradas(i, :) - centros(clase, :)))^2;
end