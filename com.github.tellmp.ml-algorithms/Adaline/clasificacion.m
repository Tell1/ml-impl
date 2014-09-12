function clases = clasificacion (W, b, entradas)

%Función que clasifica unos datos de entrada según una matriz de pesos y
%array de bias ya entrenados.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

clases = zeros(1,size(entradas,1));

for i=1:size(entradas, 1),
    y = funcionActivacion(entradas(i, :), W, b);
    clases(i) = round(y);
end