function [ prec ] = calc_precision( inputs, red )

% Funci�n que calcula la precisi�n de la proyecci�n del SOM.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013
 
    distancia = 100000000000000000000;
    indexBMU = 0;
    errDist = 0;
        
        for n = 1 : size(inputs, 1),
        % obtenci�n de la BMU
        
        for j=1:size(red.W, 1),
           d = norm(red.W(j, :) - inputs(n,:));
           if d < distancia
               distancia = d;
               indexBMU = j;
           end
        end

        errDist = errDist + norm(inputs(n) - red.W(indexBMU, :));
        end
    
    prec = errDist / size(inputs, 1);

end

