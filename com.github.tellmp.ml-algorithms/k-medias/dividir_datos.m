function [trainInputs, trainTargets, testInputs, testTargets] = dividir_datos(entradas, objetivos, parametros)

%Función que divide los datos de entrada en 2 conjuntos: entrenamiento,
% y test.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

%division del conjunto de datos
trainVector = rand(1, size(entradas, 1));

trainIndex = trainVector < parametros.trainRatio;
testIndex = (trainVector > parametros.trainRatio);

trainInputs = entradas(trainIndex, :);
testInputs = entradas(testIndex, :);

trainTargets = objetivos(trainIndex, :);
testTargets = objetivos(testIndex, :);