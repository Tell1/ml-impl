function clases = clasificacion (red, entradas, parametros)

%Funci�n que clasifica unos datos de entrada seg�n una matriz de pesos y
%array de bias ya entrenados.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

clases = zeros(size(entradas, 1), size(red.W, 2));

for i=1:size(entradas, 1),
    %salida de la primera capa (capa oculta)
    for n=1:parametros.m,
        y_capaOculta(n) = exp(-(parametros.m/red.distancia^2) * (norm(entradas(i, :) - red.centros(n, :))^2));
    end

    %salida de la �ltima capa
    for k=1:size(red.W, 2),
        y_capaSalida(k) = round(funcionActivacion(y_capaOculta, red.W(:, k), red.b(k)));
    end
    clases(i, :) = y_capaSalida;
end