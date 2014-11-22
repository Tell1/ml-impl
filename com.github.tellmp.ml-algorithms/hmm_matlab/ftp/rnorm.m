% function Y=rnorm(X);
% 
% Row normalization. Divides each row by the sum of elements in that
% row, and returns same size matrix. If the sum is 0, then replaces row by
% 1/K, where K is the number of % columns.  

function Y=rnorm(X);

[n m]=size(X);
Y=zeros(n,m);

sumX=(rsum(X)==0)';

z=find(sumX);
nz=find(~sumX);

for i=nz
  s=sum(X(i,:));
  Y(i,:)=X(i,:)/s;
end;

for i=z
  Y(i,:)=ones(1,m)/m;
end;