function clases = adaline(entradas, objetivos, parametros)

%Función que entrena un adaline y lo usa para clasificar unos
%ejemplos.
%Los datos del struct de parámetros son los siguientes:
% - maxIter: número de iteraciones.
% - trainRatio: porcentaje del conjunto de datos de entrada que se va a
%               usar para entrenar. El resti (1-trainRatio) será usado como
%               test.
% - fAprendizaje: factor de aprendizaje. En teoría, debería ser un valor
%                 entre 0 y 1.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

disp('Iniciando Adaline')

%división de los datos
[trainInputs, trainTargets, testInputs, testTargets] = dividir_datos(entradas, objetivos, parametros);

disp('Datos divididos correctamente')

%configuración
red = configuracion(trainInputs, trainTargets, parametros);

disp('Red configurada correctamente')

%entrenamiento
red = entrenamiento(red, trainInputs, trainTargets, parametros);

disp('Red entrenada correctamente')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clasificación de los datos de test
for i=1:size(testTargets, 2)
    testSal(i, :) = clasificacion (red.W(:, i), red.b(i), testInputs);
    testClases = testSal';
    
    %matriz de confusión test
    fprintf(1,'Matriz de confusión de los datos de TEST, salida %d', i)
    [matriz_confusion_test,order_test] = confusionmat(testTargets(:, i), testClases(:, i))

end

disp('Porcentaje bien calculadas, datos de TEST')
cont = 0;
for z=1:size(testClases, 1)
   if ~isequal(testTargets(z, :), testClases(z, :))
       cont = cont + 1;
   end
end
porcentajeTest = 100 - (cont/size(testTargets,1))*100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clasificación
for i=1:size(objetivos, 2)
    sal(i, :) = clasificacion (red.W(:, i), red.b(i), entradas);
    clases = sal';
    
    %matriz de confusión
    fprintf(1,'Matriz de confusión, salida %d\n', i)
    [matriz_confusion,order] = confusionmat(objetivos(:, i), clases(:, i))
    
end

disp('Porcentaje bien calculadas')
cont = 0;
for z=1:size(clases, 1)
   if ~isequal(objetivos(z, :), clases(z, :))
       cont = cont + 1;
   end
end
porcentaje = 100 - (cont/size(objetivos,1))*100