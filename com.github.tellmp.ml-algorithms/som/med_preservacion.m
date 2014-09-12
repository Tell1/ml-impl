function [ err ] = med_preservacion( inputs, red )

% Función que calcula la preservación de la topología del SOM.
% Adrián González Duarte - CNE 2012-2013
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013
    
    indexBMU1 = 0;
    indexBMU2 = 0;
    err = 0;
        
    for n = 1 : size(inputs, 1),
        distancia = 100000000000000000000;
        % obtención de la BMU
        for j=1:size(red.W, 1),
           d = norm(red.W(j, :) - inputs(n,:));
           if d < distancia
               distancia = d;
               indexBMU1 = j;
           end
        end
        
        distancia = 100000000000000000000;
        
        % obtención de la 2a BMU
        for j=1:size(red.W, 1),
           d = norm(red.W(j, :) - inputs(n,:));
           if (d < distancia && indexBMU1 ~= j)
               distancia = d;
               indexBMU2 = j;
           end
        end

        % si no son adyacentes, es decir, si la resta es distinta de 1, se
        % suma 1
        if(abs(indexBMU1 - indexBMU2) ~= 1)
            err = err + 1;
        end

    end
    
    %división por el número de datos
    err = err / size(inputs, 1);

end

