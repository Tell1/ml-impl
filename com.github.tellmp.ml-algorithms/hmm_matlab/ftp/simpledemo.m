
% A simple demo of the HMM code

load X;

% plot the time series data

plot(X,'o');

pause(5);

% get help on the function hmm

help hmm;

pause(2);

% run a 4-state HMM on the data

[Mu,Cov,P,Pi,LL]=hmm(X,200,4);

% plot learning curve

plot(LL/200);
xlabel('iterations of EM');
ylabel('log probability per event');

pause(2); 

% plot means of learned HMM centers on top of data

plot(X,'o');
hold on;
plot((ones(4,1)*[0 200])',(Mu*[1 1])','r-');
hold off;