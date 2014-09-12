function clases = perceptron_multicapa(entradas, objetivos, parametros)

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
% - numClases: número de clases diferentes que hay en el conjunto de datos.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

disp('Iniciando el perceptrón multicapa')

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

for z=1:size(testClases, 2)
	%matriz de confusión test
	fprintf(1,'Matriz de confusión de los datos de TEST, salida %d', z)
	[matriz_confusion_test,order_test] = confusionmat(testTargets(:, z), testClases(:, z))
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
clases = clasificacion (red, entradas, parametros);

for z=1:size(testClases, 2)
	%matriz de confusión
	fprintf(1,'Matriz de confusión, salida %d', z)
	[matriz_confusion,order] = confusionmat(objetivos(:, z), clases(:, z))
end


%cálculo del porcentaje
disp('Porcentaje bien calculadas')
cont = 0;
for z=1:size(clases, 1)
   if ~isequal(objetivos(z, :), clases(z, :))
       cont = cont + 1;
   end
end
porcentaje = 100 - (cont/size(objetivos,1))*100