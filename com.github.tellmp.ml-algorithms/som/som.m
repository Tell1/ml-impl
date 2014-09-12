function clases = som(entradas, objetivos, parametros)

%Función que entrena un k-medias y lo usa para clasificar unos
%ejemplos.
%Los datos del struct de parámetros son los siguientes:
% - maxIter: número de iteraciones.
% - trainRatio: porcentaje del conjunto de datos de entrada que se va a
%               usar para entrenar.
% - fAprendizaje: factor de aprendizaje inicial.
% - m: número de neuronas del som.
% - numCentros: número de clases.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013


%división de los datos
[trainInputs, trainTargets, testInputs, testTargets] = dividir_datos(entradas, objetivos, parametros);

%configuración
red = configuracion(trainInputs, trainTargets, parametros);

%entrenamiento
red = entrenamiento(red, trainInputs, trainTargets, parametros);

disp('Precisión de la proyección, datos de TEST')
precTrain = calc_precision( trainInputs, red )
disp('Preservación de la topología, datos de TEST')
errClas = med_preservacion( trainInputs, red )

%clasificación
clases = clasificacion (red, entradas, parametros);

disp('Precisión de la proyección')
precClas = calc_precision( entradas, red )
disp('Preservación de la topología')
errClas = med_preservacion( entradas, red )