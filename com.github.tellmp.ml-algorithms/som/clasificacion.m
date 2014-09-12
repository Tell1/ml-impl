function clases = clasificacion (red, entradas, parametros)

%Funci�n que clasifica unos datos de entrada seg�n unos pesos ya
%entrenados
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

[IDX,C] = kmeans(red.W,parametros.numCentros);

for i=1:size(entradas, 1),
    %buscar la BMU
    distancia = norm(red.W(1, :) - entradas(i, :));
    indexBMU = 1;
    
    for j=2:size(red.W, 1),
       d = norm(red.W(j, :) - entradas(i, :));
       if d < distancia
           distancia = d;
           indexBMU = j;
       end
    end

    distancia = norm(red.W(indexBMU, :) - C(1, :));
    ind = 1;
    for z=2:size(C,1)
        d = norm(red.W(indexBMU, :) - C(z, :));
        if d < distancia
            distancia = d;
            ind = z;
        end
    end
    clases(i) = ind;
end
clases = clases';