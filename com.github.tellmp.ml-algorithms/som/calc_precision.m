function [ prec ] = calc_precision( inputs, red )

% Función que calcula la precisión de la proyección del SOM.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013
 
    distancia = 100000000000000000000;
    indexBMU = 0;
    errDist = 0;
        
        for n = 1 : size(inputs, 1),
        % obtención de la BMU
        
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

