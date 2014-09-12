function clases = som(entradas, objetivos, parametros)

%Funci�n que entrena un k-medias y lo usa para clasificar unos
%ejemplos.
%Los datos del struct de par�metros son los siguientes:
% - maxIter: n�mero de iteraciones.
% - trainRatio: porcentaje del conjunto de datos de entrada que se va a
%               usar para entrenar.
% - fAprendizaje: factor de aprendizaje inicial.
% - m: n�mero de neuronas del som.
% - numCentros: n�mero de clases.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013


%divisi�n de los datos
[trainInputs, trainTargets, testInputs, testTargets] = dividir_datos(entradas, objetivos, parametros);

%configuraci�n
red = configuracion(trainInputs, trainTargets, parametros);

%entrenamiento
red = entrenamiento(red, trainInputs, trainTargets, parametros);

disp('Precisi�n de la proyecci�n, datos de TEST')
precTrain = calc_precision( trainInputs, red )
disp('Preservaci�n de la topolog�a, datos de TEST')
errClas = med_preservacion( trainInputs, red )

%clasificaci�n
clases = clasificacion (red, entradas, parametros);

disp('Precisi�n de la proyecci�n')
precClas = calc_precision( entradas, red )
disp('Preservaci�n de la topolog�a')
errClas = med_preservacion( entradas, red )