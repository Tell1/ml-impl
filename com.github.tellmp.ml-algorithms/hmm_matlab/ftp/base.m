% function v=base(k,m,d)
%
% returns a vector of length d with (k base m) +1
%
% for example base(29,5,4)=[1 2 1 5]
% (if d omitted returns shortest vector)

function v=base(k,m,d)

if nargin<3
  d=fix(log(k)/log(m)+1);
end;  

mm=m.^(d-1:-1:0);

for i=1:d
  v(i)=fix(k/mm(i));
  k=k-mm(i)*v(i);
end;

v=v+1;
