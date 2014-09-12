function clases = clasificacion (red, entradas, parametros)

%Funci�n que clasifica unos datos de entrada seg�n una matriz de pesos y
%array de bias ya entrenados.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

clases = zeros(size(entradas, 1), size(red.W_2, 2));

for i=1:size(entradas, 1),
    %salida de la primera capa (capa oculta)
    for n=1:parametros.m,
        y_capaOculta(n) = funcionActivacion(entradas(i, :), red.W_1(:, n), red.b_1(n), 0);
    end

    %salida de la �ltima capa
    for k=1:size(red.W_2, 2),
        y = funcionActivacion(y_capaOculta, red.W_2(:, k), red.b_2(k), 1);
        if parametros.numClases > 2
            for j=1:parametros.numClases
               d = j*(1/parametros.numClases);
               if y <= d
                   if j == 1
                       y_capaSalida(k) = 0;
                       break;
                   else if j == parametros.numClases
                          y_capaSalida(k) = 1;
                          break;
                       else
                           y_capaSalida(k) = (d + ((j-1)*(1/parametros.numClases)))/2;
                           break;
                       end
                   end

               end
            end
        else
            y_capaSalida(k) = round(y);
        end
    end
    clases(i, :) = y_capaSalida;
end