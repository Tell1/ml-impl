function clases = k_medias(entradas, objetivos, parametros)

%Funci�n que entrena un k-medias y lo usa para agrupar unos
%ejemplos.
%Los datos del struct de par�metros son los siguientes:
% - maxIter: n�mero de iteraciones.
% - trainRatio: porcentaje del conjunto de datos de entrada que se va a
%               usar para entrenar.
% - numCentros: n�mero de centros que va a buscar el algoritmo
% - kMeansPP: un valor de 0 indica que se va a usar el k-medias normal.
%             Cualquier otro valor indica que se va a usar la variante 
%             k-medias++.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

%divisi�n de los datos
[trainInputs, trainTargets, testInputs, testTargets] = dividir_datos(entradas, objetivos, parametros);

%configuraci�n
red = configuracion(trainInputs, trainTargets, parametros);

%entrenamiento
red = entrenamiento(red, trainInputs, trainTargets, parametros);

disp('Los centros encontrados son: ')
red.centros

%clasificaci�n y error de los datos de test
clasesTest = clasificacion (red, testInputs, parametros);
%medida de error
disp('Medida del error, datos de TEST:')
errorTest = medida_error(clasesTest, testInputs, red.centros);
errorTest

%clasificaci�n
clases = clasificacion (red, entradas, parametros);

%medida de error
disp('Medida del error:')
error = medida_error(clases, entradas, red.centros);
error

%tama�o de las regiones de voronoi
disp('Tama�o de las regiones de Voronoi:')
tam = regiones_voronoi(clases, red.centros);
tam

