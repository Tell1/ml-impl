function [ err ] = med_preservacion( inputs, red )

% Funci�n que calcula la preservaci�n de la topolog�a del SOM.
% Adri�n Gonz�lez Duarte - CNE 2012-2013
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013
    
    indexBMU1 = 0;
    indexBMU2 = 0;
    err = 0;
        
    for n = 1 : size(inputs, 1),
        distancia = 100000000000000000000;
        % obtenci�n de la BMU
        for j=1:size(red.W, 1),
           d = norm(red.W(j, :) - inputs(n,:));
           if d < distancia
               distancia = d;
               indexBMU1 = j;
           end
        end
        
        distancia = 100000000000000000000;
        
        % obtenci�n de la 2a BMU
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
    
    %divisi�n por el n�mero de datos
    err = err / size(inputs, 1);

end

