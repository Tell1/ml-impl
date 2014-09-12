function clases = perceptron_multicapa(entradas, objetivos, parametros)

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
% - numClases: n�mero de clases diferentes que hay en el conjunto de datos.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

disp('Iniciando el perceptr�n multicapa')

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

for z=1:size(testClases, 2)
	%matriz de confusi�n test
	fprintf(1,'Matriz de confusi�n de los datos de TEST, salida %d', z)
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

%clasificaci�n
clases = clasificacion (red, entradas, parametros);

for z=1:size(testClases, 2)
	%matriz de confusi�n
	fprintf(1,'Matriz de confusi�n, salida %d', z)
	[matriz_confusion,order] = confusionmat(objetivos(:, z), clases(:, z))
end


%c�lculo del porcentaje
disp('Porcentaje bien calculadas')
cont = 0;
for z=1:size(clases, 1)
   if ~isequal(objetivos(z, :), clases(z, :))
       cont = cont + 1;
   end
end
porcentaje = 100 - (cont/size(objetivos,1))*100