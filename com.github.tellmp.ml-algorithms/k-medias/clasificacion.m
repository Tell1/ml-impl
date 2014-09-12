function clases = clasificacion (red, entradas, parametros)

%Función que clasifica unos datos de entrada según unos centros ya
%entrenados
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

clases = zeros(size(entradas, 1), 1);

for i=1:size(entradas, 1),    
    distancia = norm(entradas(i, :) - red.centros(1, :));
    j = 1;
    centroMasCercano = j;        
    for j=2:parametros.numCentros
        newDistancia = norm(entradas(i, :) - red.centros(j, :));
        if newDistancia < distancia
            distancia = newDistancia;
            centroMasCercano = j;
        end
    end
    clases(i, :) = centroMasCercano;
end