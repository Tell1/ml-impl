function clases = clasificacion (W, b, entradas)

%Función que clasifica unos datos de entrada según una matriz de pesos y
%array de bias ya entrenados.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

for i=1:size(entradas, 1),
    clases(i) = funcionActivacion(entradas(i, :), W, b);
    clases = clases';
end