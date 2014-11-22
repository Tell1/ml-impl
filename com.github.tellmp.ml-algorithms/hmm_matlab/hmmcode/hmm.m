function [Y,states] = hmm(N,T,p0,C,R)
%[Y,states] = hmm(N,T,p0,C,R)
%
% Runs an HMM as a generative model generating N output vectors
%
% N is a scalar or vector saying how many outputs to generate
% (if vector, multiple sequences are generated)
%
% T(i,j) is the probability of going to j next if you are now in i
% p0(j)  is the probability of starting in state j
% C(q,j) is the q^th coordinate of the j^th state mean vector
% R(q,r) is the covariance between the q and r coordinate for observations
%        or a scalar if R is a multiple of the identity matrix	
%
% Y is a matrix of observations, one per column
%   (or cell array of matrices if many sequences)
%

[pp,kk] = size(C);
if(all(size(R)==1)) R=R*eye(pp,pp); end
[pp2,pp3] = size(R); assert(pp2==pp & pp3==pp);
p0 = cumsum(p0);
if(~issparse(T)) T = cumsum(T,2); end
    % don't do the cumsum beforehand if T is big but sparse to avoid
    % allocating memory for a full sized T

nseqs=length(N);
if(nseqs==1) 
  [Y,states]=genseq(N,T,p0,C,R);
else
  Y=cell(nseqs,1); states=cell(nseqs,1);
  for nn=1:nseqs
    [Y{nn},states{nn}]=genseq(N(nn),T,p0,C,R);
  end
end

function [yy,st]=genseq(len,T,p0,C,R)

st=zeros(1,len);
ff=find(rand<p0); 
st(1) = ff(1);

for tt=2:len
  if(~issparse(T))
    ff=find(rand<T(st(tt-1),:)); 
  else
    ff=find(rand<cumsum(T(states(tt-1),:)));
  end
  st(tt) = ff(1);    
end

yy = C(:,st)+sqrtm(R)*randn(size(C,1),len);
