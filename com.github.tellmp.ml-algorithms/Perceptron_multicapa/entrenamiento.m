function red = entrenamiento(red, trainInputs, trainTargets, parametros)

%Entrenamiento de la red.
%Devuelve la matriz de pesos y el array de bias ya entrenados.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

%Construcción de la matriz W
red.W_1 = zeros(size(trainInputs, 2), parametros.m);
red.W_2 = zeros(parametros.m, size(trainTargets, 2));

%Inicialización
red.W_1 = rand(size(red.W_1));
red.W_2 = rand(size(red.W_2));

W_1_ant = zeros(size(trainInputs, 2), parametros.m);
W_2_ant = zeros(parametros.m, size(trainTargets, 2));

%Construcción del array de bias
red.b_1 = zeros(1, parametros.m);
red.b_2 = zeros(1, size(trainTargets, 2));

%Inicialización
red.b_1 = rand(size(red.b_1));
red.b_2 = rand(size(red.b_2));

for i=1:size(red.b_1)
    red.b_1(i) = randi([0 ,1]);
end

for i=1:size(red.b_2)
    red.b_2(i) = randi([0 ,1]);
end

b_1_ant = zeros(1, parametros.m);
b_2_ant = zeros(1, size(trainTargets, 2));

W_1_ant = red.W_1;
W_2_ant = red.W_2;
b_1_ant = red.b_1;
b_2_ant = red.b_2;

maxFailures = 20;
contFailures = 0;

for i=1:parametros.maxIter,
    
    %selección de una fila aleatoria
    randomIndex = randi([1, size(trainInputs, 1)]);
    randomRowInput = trainInputs(randomIndex, :);
    randomRowTarget = trainTargets(randomIndex, :);
    
    y_capaOculta = zeros(1, parametros.m);
    y_capaSalida = zeros(1, size(trainTargets, 2));
    
    %%%%%% FASE FEEDFORWARD %%%%%%%
    
    %salida de la primera capa (capa oculta)
    for n=1:parametros.m,
        y_capaOculta(n) = funcionActivacion(randomRowInput, red.W_1(:, n), red.b_1(n), 0);
    end

    %salida de la última capa
    for k=1:size(trainTargets, 2),
        y_capaSalida(k) = funcionActivacion(y_capaOculta, red.W_2(:, k), red.b_2(k), 1);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%% FASE FEEDBACK %%%%%%%%%
    
    e = zeros(1, size(trainTargets, 2));
    
    %reajuste de los pesos que unen la capa oculta con la de salida
    for j=1:size(trainTargets, 2),
        %cálculo del error
        e(j) = (randomRowTarget(j) - y_capaSalida(j));
        
        %adaptación de los pesos
        for n=1:parametros.m,
            [red.W_2(n, j), red.b_2(j), W_2_ant(n, j), b_2_ant(j)] = reglaAprendizaje (red.W_2(n, j), red.b_2(j), y_capaSalida(j), e(j), parametros.fAprendizaje, 0, 0, W_2_ant(n, j), b_2_ant(j), parametros.fMomento);
        end
    end
    
    %reajuste de los pesos que unen la capa de entrada con la oculta
    for n=1:parametros.m,
        %término delta, con los errores producidos en la capa de salida
        sum = 0;
        for j=1:size(trainTargets, 2),
            sum = sum + (W_2_ant(n,j) * e(j) * (y_capaSalida(j) * (1 - y_capaSalida(j))));
        end
        sum = (y_capaOculta(n) * (1 - y_capaOculta(n))) * sum;
        
        %adaptación de los pesos
        for z=1:size(randomRowInput,2),
            [red.W_1(z,n), red.b_1(n), W_1_ant(z,n), b_1_ant(n)] = reglaAprendizaje(red.W_1(z, n), red.b_1(n), randomRowInput(z), 0, parametros.fAprendizaje, 1, sum, W_1_ant(z,n), b_1_ant(n), parametros.fMomento);
        end
    end
    
end
