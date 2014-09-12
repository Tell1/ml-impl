function red = entrenamiento(red, trainInputs, trainTargets, parametros)

%Entrenamiento de la red.
%Devuelve la matriz de pesos y el array de bias ya entrenados.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

%Construcción de la matriz W
red.W = zeros(size(trainInputs, 2),size(trainTargets, 2));

%Inicialización
red.W = rand(size(red.W));

%Construcción del array de bias
red.b = zeros(1, size(trainTargets, 2));

%Inicialización
red.b = rand(size(red.b));    %inicializamos

for i=1:parametros.maxIter,
    
    %selección de una fila aleatoria
    randomIndex = randi([1, size(trainInputs, 1)]);
    randomRowInput = trainInputs(randomIndex, :);
    randomRowTarget = trainTargets(randomIndex, :);
    
    %Entrenamos un perceptrón por cada una de las columnas de la salida, ya
    %que un perceptrón sólo puede dar una única salida.
    for j=1:size(trainTargets, 2),
        %función de activación
        y = funcionActivacion(randomRowInput, red.W(:, j), red.b(j));
                
        %cálculo del error cometido
        e = (randomRowTarget(j) - y);
        
        %adaptación de los pesos
        [red.W(:, j), red.b(j)] = reglaAprendizaje (red.W(:, j), red.b(j), randomRowInput, e, parametros.fAprendizaje);
        
    end
    
end