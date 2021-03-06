function red = entrenamiento(red, trainInputs, trainTargets, parametros)

%Entrenamiento de la red.
%Devuelve la matriz de pesos y el array de bias ya entrenados.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

%Construcci�n de la matriz W
red.W = zeros(parametros.m, size(trainTargets, 2));

%Inicializaci�n
red.W = rand(size(red.W));

W_ant = zeros(parametros.m, size(trainTargets, 2));

%Construcci�n del array de bias
red.b = zeros(1, size(trainTargets, 2));

%Inicializaci�n
red.b = rand(size(red.b));

b_ant = zeros(1, size(trainTargets, 2));

W_ant = red.W;
b_ant = red.b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Selecci�n inicial de centros

red.centros = zeros(parametros.m, size(trainInputs, 2));

randomIndex = randi([1, size(trainInputs, 1)]);
randomRowInput = trainInputs(randomIndex, :);
red.centros(1, :) =  randomRowInput;
red.distancia = 0;

for n=2:parametros.m
    randomIndex2 = randi([1, size(trainInputs, 1)]);
    while randomIndex2 == randomIndex,
        randomIndex2 = randi([1, size(trainInputs, 1)]);
    end

    red.centros(n, :) = trainInputs(randomIndex2, :);
    
    %c�lculo de la distancia
    for z=1:size(red.centros, 1)
        d = norm(red.centros(z, :) - red.centros(n, :));
        
        %guardamos la distancia mayor
        if d > red.distancia
            red.distancia = d;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% APRENDIZAJE DE LOS CENTROS %%%%%%%
for i=1:parametros.maxIter,
    
    %selecci�n de una fila aleatoria
    randomIndex = randi([1, size(trainInputs, 1)]);
    randomRowInput = trainInputs(randomIndex, :);
    
    %b�squeda del centro m�s cercano
    dist = 1000000000000;
    for j=1:size(red.centros, 1)
       d =  norm(red.centros(j, :) - randomRowInput);
       if d < dist
           dist = d;
           index = j;
       end
    end
    
    %modificaci�n del centro
    red.centros(index, :) =  red.centros(index, :)+(parametros.fAprendizajeCentros*(red.centros(index, :) - randomRowInput));
    
    %b�squeda de la m�xima distancia
    for z=1:size(red.centros, 1)
        for k=1:size(red.centros,1)
            if k ~= z
               d = norm(red.centros(z, :) - red.centros(n, :));
        
                %guardamos la distancia mayor
                if d > red.distancia
                    red.distancia = d;
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% APRENDIZAJE DE LOS PESOS %%%%%%%
for i=1:parametros.maxIter,
    
    %selecci�n de una fila aleatoria
    randomIndex = randi([1, size(trainInputs, 1)]);
    randomRowInput = trainInputs(randomIndex, :);
    randomRowTarget = trainTargets(randomIndex, :);
    
    y_capaOculta = zeros(1, parametros.m);
    y_capaSalida = zeros(1, size(trainTargets, 2));
    
    %salida de la primera capa (capa oculta)      
    for n=1:parametros.m,
        y_capaOculta(n) = exp(-(parametros.m/red.distancia^2) * (norm(randomRowInput - red.centros(n, :))^2));
    end

    %salida de la �ltima capa
    for k=1:size(trainTargets, 2),
        y_capaSalida(k) = funcionActivacion(y_capaOculta, red.W(:, k), red.b(k));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%% FASE FEEDBACK %%%%%%%%%
    
    e = zeros(1, size(trainTargets, 2));
    
     %reajuste de los pesos que unen la capa oculta con la de salida
    for j=1:size(trainTargets, 2),
        %c�lculo del error
        e(j) = (randomRowTarget(j) - y_capaSalida(j));
        
        %adaptaci�n de los pesos
        for n=1:parametros.m,
            [red.W(n, j), red.b(j), W_ant(n, j), b_ant(j)] = reglaAprendizaje (red.W(n, j), red.b(j), y_capaSalida(j), e(j), parametros.fAprendizaje, W_ant(n, j), b_ant(j), parametros.fMomento);
        end
    end
    
end
