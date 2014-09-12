function red = entrenamiento(red, trainInputs, trainTargets, parametros)

%Entrenamiento de la red.
%Devuelve la matriz de pesos adaptada.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

% Inicializaci�n
red.W = rand(size(red.W));

sigma_cero = size(red.W, 1)/2;
fAprendizajeInicial = parametros.fAprendizaje;
anchuraVecindario = sigma_cero;

for i=1:parametros.maxIter,
    
    % selecci�n de una fila
    randomIndex = randi([1, size(trainInputs, 1)]);
    randomRowInput = trainInputs(randomIndex, :);
    
    distancia = 100000000000000000000;
    indexBMU = 0;
        
    % obtenci�n de la BMU
    for j=1:size(red.W, 1),
       d = norm(red.W(j, :) - randomRowInput);
       if d < distancia
           distancia = d;
           indexBMU = j;
       end
    end
    
    % adaptaci�n de los pesos
    e = randomRowInput - red.W(indexBMU, :);
    for k=1:size(red.W, 1)
        d = norm(red.W(k, :) - randomRowInput);
        v = funcionVecindario(d, anchuraVecindario);
        red.W(k, :) = reglaAprendizaje(0, e, parametros.fAprendizaje, v, red.W(k, :));
    end
    
    % adaptaci�n del factor de aprendizaje
    red.fAprendizaje = fAprendizajeInicial * exp(-i/parametros.maxIter);
    
    % c�lculo de la anchura del vecindario
    anchuraVecindario = sigma_cero * (exp(-i/(parametros.maxIter/log(sigma_cero))));
    
    
end