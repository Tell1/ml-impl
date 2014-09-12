function v = funcionVecindario(d, s)

% Función que va a devolver un valor según la distancia y el radio del
% vecindario. Cuanto mayor sea la distancia, más pequeño será este valor, y
% cuanto menor sea el radio, más decrecerá el valor con la distancia.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

v = exp(-((d^2)/(2*s^2)));