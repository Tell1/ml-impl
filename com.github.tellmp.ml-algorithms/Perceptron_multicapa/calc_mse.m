function [ mse ] = calc_mse( trainTargets, y_capaSalida )

%Calculo del MSE - J = 1/2 * sum_t(target - output)^2
% Adrián González Duarte 
% Tell Müller-Pettenpohl
% CNE 2012-2013

    err = 0;
    for k = 1 : size(trainTargets, 2),
        err = err + (trainTargets(k) - y_capaSalida(k))^2;
    end
    mse = 1/2 * err;
end

