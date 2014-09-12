function clases = clasificacion (red, entradas, parametros)

%Funci�n que clasifica unos datos de entrada seg�n unos centros ya
%entrenados
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
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