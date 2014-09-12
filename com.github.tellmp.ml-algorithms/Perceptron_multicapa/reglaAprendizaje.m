function [D, threshold, W_ant, b_ant] = reglaAprendizaje (W, b, x, e, fAprendizaje, tipo, sum, W_anterior, b_anterior, fMomento)

%Recalcula la matriz de pesos y el array de bias.
%Para el recálculo del bias usamos una extensión de la regla de
%aprendizaje, en la que el nuevo bias será el anterior más el error.
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

W_ant = W;
b_ant = b;

switch tipo,
    case 0 %caso de que sea de la última capa
        delta = e * (x * (1 - x));
        
    case 1
        delta = sum;
end

%cálculo del momento
momento_pesos = fMomento*(W-W_anterior);
momento_bias = fMomento*(b - b_anterior);

n = (fAprendizaje*delta*x)';
n_bias = (fAprendizaje*delta)';

D = W + n + momento_pesos;

threshold = b + n_bias + momento_bias;
