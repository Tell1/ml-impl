function [Y,states] = hmmx(N,T,p0,H)
%[Y,states] = hmmx(N,T,p0,H)
%
% Runs an HMM as a generative model generating N output symbols
%
% N is a scalar or vector saying how many outputs to generate
% (if vector, multiple sequences are generated)
%
% T(i,j) is the probability of going to j next if you are now in i
% p0(j)  is the probability of starting in state j
% H(m,j) is the probability of emitting symbol m if you are in state j
%
% Y is a vector (or cell array of vectors if many sequences) of integers
%

[M,kk] = size(H);
H = cumsum(H,1);
p0 = cumsum(p0);
if(~issparse(T)) T = cumsum(T,2); end
    % don't do the cumsum beforehand if T is big but sparse to avoid
    % allocating memory for a full sized T

nseqs=length(N);
if(nseqs==1) 
  [Y,states]=genseq(N,T,p0,H);
else
  Y=cell(nseqs,1); states=cell(nseqs,1);
  for nn=1:nseqs
    [Y{nn},states{nn}]=genseq(N(nn),T,p0,H);
  end
end

function [yy,st]=genseq(len,T,p0,H)

st=zeros(1,len);
yy = zeros(1,len);

ff=find(rand<p0); 
st(1) = ff(1);
ff=find(rand<H(:,st(1)));
yy(1) = ff(1);

for tt=2:len
  if(~issparse(T))
    ff=find(rand<T(st(tt-1),:)); 
  else
    ff=find(rand<cumsum(T(states(tt-1),:)));
  end
  st(tt) = ff(1);    
  ff=find(rand<H(:,st(tt)));
  yy(tt) = ff(1);
end


