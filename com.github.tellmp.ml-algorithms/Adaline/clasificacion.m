function clases = clasificacion (W, b, entradas)

%Funci�n que clasifica unos datos de entrada seg�n una matriz de pesos y
%array de bias ya entrenados.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

clases = zeros(1,size(entradas,1));

for i=1:size(entradas, 1),
    y = funcionActivacion(entradas(i, :), W, b);
    clases(i) = round(y);
end