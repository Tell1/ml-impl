function clases = base_radial(entradas, objetivos, parametros)

%Función que entrena un perceptrón multicapa y lo usa para clasificar unos
%ejemplos.
%Los datos del struct de parámetros son los siguientes:
% - maxIter: número de iteraciones.
% - trainRatio: porcentaje del conjunto de datos de entrada que se va a
%               usar para entrenar. El resto (1-trainRatio), se usará como
%               test.
% - fAprendizaje: factor de aprendizaje. En teoría, debería ser un valor
%                 entre 0 y 1.
% - m: número de neuronas en la capa oculta.
% - fMomento: factor utilizado para calcular el momento.
% - fAprendizajeCentros: factor utilizado para determinar cuánto se mueven
%                        los centros en el aprendizaje.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

disp('Iniciando redes de funciones de base radial')

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
testClases = clasificacion (red, testInputs, parametros);

%matriz de confusión test
disp('Matriz de confusión de los datos de TEST')
[matriz_confusion_test,order_test] = confusionmat(testTargets, testClases)

%cálculo del porcentaje test
disp('Porcentaje bien calculadas datos de TEST')
porcentajeTest = 100 - (sum(testTargets~=testClases)/size(testTargets,1))*100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clasificación
clases = clasificacion (red, entradas, parametros);

%matriz de confusión
[matriz_confusion,order] = confusionmat(objetivos, clases)

%cálculo del porcentaje
disp('Porcentaje bien calculadas')
porcentaje = 100 - (sum(objetivos~=clases)/size(objetivos,1))*100