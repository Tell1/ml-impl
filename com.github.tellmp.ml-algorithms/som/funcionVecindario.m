function v = funcionVecindario(d, s)

% Funci�n que va a devolver un valor seg�n la distancia y el radio del
% vecindario. Cuanto mayor sea la distancia, m�s peque�o ser� este valor, y
% cuanto menor sea el radio, m�s decrecer� el valor con la distancia.
% Adri�n Gonz�lez Duarte 
% Tell M�ller-Pettenpohl
% CNE 2012-2013

v = exp(-((d^2)/(2*s^2)));