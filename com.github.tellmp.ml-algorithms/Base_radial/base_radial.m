function clases = base_radial(entradas, objetivos, parametros)

%Funci�n que entrena un perceptr�n multicapa y lo usa para clasificar unos
%ejemplos.
%Los datos del struct de par�metros son los siguientes:
% - maxIter: n�mero de iteraciones.
% - trainRatio: porcentaje del conjunto de datos de entrada que se va a
%               usar para entrenar. El resto (1-trainRatio), se usar� como
%               test.
% - fAprendizaje: factor de aprendizaje. En teor�a, deber�a ser un valor
%                 entre 0 y 1.
% - m: n�mero de neuronas en la capa oculta.
% - fMomento: factor utilizado para calcular el momento.
% - fAprendizajeCentros: factor utilizado para determinar cu�nto se mueven
%                        los centros en el aprendizaje.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

disp('Iniciando redes de funciones de base radial')

%divisi�n de los datos
[trainInputs, trainTargets, testInputs, testTargets] = dividir_datos(entradas, objetivos, parametros);

disp('Datos divididos correctamente')

%configuraci�n
red = configuracion(trainInputs, trainTargets, parametros);

disp('Red configurada correctamente')

%entrenamiento
red = entrenamiento(red, trainInputs, trainTargets, parametros);

disp('Red entrenada correctamente')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clasificaci�n de los datos de test
testClases = clasificacion (red, testInputs, parametros);

%matriz de confusi�n test
disp('Matriz de confusi�n de los datos de TEST')
[matriz_confusion_test,order_test] = confusionmat(testTargets, testClases)

%c�lculo del porcentaje test
disp('Porcentaje bien calculadas datos de TEST')
porcentajeTest = 100 - (sum(testTargets~=testClases)/size(testTargets,1))*100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clasificaci�n
clases = clasificacion (red, entradas, parametros);

%matriz de confusi�n
[matriz_confusion,order] = confusionmat(objetivos, clases)

%c�lculo del porcentaje
disp('Porcentaje bien calculadas')
porcentaje = 100 - (sum(objetivos~=clases)/size(objetivos,1))*100