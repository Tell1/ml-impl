function y = funcionActivacion(x, W, b, tipo)

%Funci�n que calcula la funci�n de activaci�n de un perceptr�n multicapa.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

%c�lculo de la suma ponderada
a = x * W + b;

%c�lculo de la funci�n de activaci�n

    y = 1.0 / (1.0 + exp(-a));


end