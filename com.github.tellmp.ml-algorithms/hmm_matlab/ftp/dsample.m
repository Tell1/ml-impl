% function x = dsample(p);
% 
% sample from a discrete distribution
%
% p is a K x 1 or 1 x K probability vector (positive, sums to 1)
% x is same size as p with 1 in position i with probabiliy p(i) and 0 elsewhere 

function x = dsample(p);

x=zeros(size(p));
f=find(rand<cumsum(p));
x(f(1))=1;



