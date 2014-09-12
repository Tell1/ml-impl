function y = funcionActivacion(x, W, b)

%Funci�n que calcula la funci�n de activaci�n de un perceptr�n simple.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

%c�lculo de la suma ponderada
a = x * W + b;

%c�lculo de la funci�n de activaci�n
if a > 0
    y = 1;
else
    y = 0;
end