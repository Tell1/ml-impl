function y = funcionActivacion(x, W, b)

%Función que calcula la función de activación de un perceptrón simple.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

%cálculo de la suma ponderada
a = x * W + b;

%cálculo de la función de activación
if a > 0
    y = 1;
else
    y = 0;
end