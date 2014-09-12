function red = entrenamiento(red, trainInputs, trainTargets, parametros)

%Entrenamiento de la red.
%Devuelve los centros encontrados.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

%Creaci�n de la matriz de grupos.
% En esta matriz se van a ir guardando los ejemplos junto con el grupo que
% les va asignando la red.
matriz_grupos = zeros(size(trainInputs, 1), (size(trainInputs, 2) + 1));

%inicializaci�n matriz de centros
red.centros = zeros(parametros.numCentros, size(trainInputs, 2));

randomIndex = randi([1, size(trainInputs, 1)]);
randomRowInput = trainInputs(randomIndex, :);
red.centros(1, :) =  randomRowInput;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(parametros.kMeansPP == 0)
%Selecci�n de centros: se seleccionan como centros iniciales algunos de los
%ejemplos, de forma aleatoria.

    for i=2:parametros.numCentros
    randomIndex2 = randi([1, size(trainInputs, 1)]);
        while randomIndex2 == randomIndex,
            randomIndex2 = randi([1, size(trainInputs, 1)]);
        end

        red.centros(i, :) = trainInputs(randomIndex2, :);
    end
    
else
%%%%% K-MEDIAS++ %%%%%%%%%%%%%%%%
% el primer centro aleatorio ya est� elegido
    cont_centros = 1;
    distribution = zeros(size(trainInputs, 1), 1);
    %recorremos los datos

    for cent=2:parametros.numCentros
        for t=1:size(trainInputs,1)

            %determinar el centro m�s cercano de los que ya hay
            distancia = norm(trainInputs(t, :) - red.centros(1, :));
            for z=2:cont_centros
               d = norm(trainInputs(t, :) - red.centros(z, :));
               if d < distancia
                   distancia = d;
               end
            end
            %distribucion de probabilidad
            distribution(t) = distancia^2;
        end

        %�ndice aleatorio
        y = randsample(size(trainInputs,1),1,true,distribution);
        red.centros(cent, :) = trainInputs(y, :);

    end
end

disp('iniciales')
red.centros

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:parametros.maxIter,
    
    for k=1:size(trainInputs, 1),
        %calcular cu�l es el centro m�s cercano al ejemplo que se est�
        %considerando
        distancia = norm(trainInputs(k, :) - red.centros(1, :));
        j = 1;
        centroMasCercano = j;        
        for j=2:parametros.numCentros
            newDistancia = norm(trainInputs(k, :) - red.centros(j, :));
            if newDistancia < distancia
                distancia = newDistancia;
                centroMasCercano = j;
            end
        end
        matriz_grupos(k, :) = [trainInputs(k, :) centroMasCercano];        
    end
    
    centrosAnt = red.centros;
    
    for k=1:parametros.numCentros
       %sacamos todos los ejemplos de cada grupo
       cont = 1;
       suma = 0;
       grupo = zeros(size(trainInputs,1), size(trainInputs, 2));
       for z=1:size(trainInputs,1)
           fila = matriz_grupos(z, :);
           if fila(size(fila,2)) == k
               fila(size(fila,2)) = []; %eliminamos la columna del grupo
               grupo(cont, :) = fila;
               cont = cont + 1;
           end
       end
	   
       %calculamos el sumatorio de todos los ejemplos del grupo
       suma = sum(grupo);
        
       %nuevo centro
       red.centros(k, :) = suma/cont;

    end
    
    %si los centros no han cambiado, salimos
    if centrosAnt == red.centros
        fprintf('Finalizado en iteraci�n %d\n', i)
        return
    end
    
end
