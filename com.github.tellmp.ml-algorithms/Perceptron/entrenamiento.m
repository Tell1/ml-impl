function red = entrenamiento(red, trainInputs, trainTargets, parametros)

%Entrenamiento de la red.
%Devuelve la matriz de pesos y el array de bias ya entrenados.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

%Construcci�n de la matriz W
red.W = zeros(size(trainInputs, 2),size(trainTargets, 2));

%Inicializaci�n
red.W = rand(size(red.W));

%Construcci�n del array de bias
red.b = zeros(1, size(trainTargets, 2));

%Inicializaci�n
red.b = rand(size(red.b));    %inicializamos

for i=1:parametros.maxIter,
    
    %selecci�n de una fila aleatoria
    randomIndex = randi([1, size(trainInputs, 1)]);
    randomRowInput = trainInputs(randomIndex, :);
    randomRowTarget = trainTargets(randomIndex, :);
    
    %Entrenamos un perceptr�n por cada una de las columnas de la salida, ya
    %que un perceptr�n s�lo puede dar una �nica salida.
    for j=1:size(trainTargets, 2),
        %funci�n de activaci�n
        y = funcionActivacion(randomRowInput, red.W(:, j), red.b(j));
                
        %c�lculo del error cometido
        e = (randomRowTarget(j) - y);
        
        %adaptaci�n de los pesos
        [red.W(:, j), red.b(j)] = reglaAprendizaje (red.W(:, j), red.b(j), randomRowInput, e, parametros.fAprendizaje);
        
    end
    
end