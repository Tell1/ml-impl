% function lik=calcmflike(X,T,mf,M,K,Mu,Cov,P,Pi);
% 
% Calculate Likelihood for Mean Field Factorial Hidden Markov Model 
% for a set of clamped sequences (Not using efficent Mth order decimation)
%
% X - NT x p data matrix
% T - length of sequences
% mf - mean field parameters
% M - number of chains 
% K - number of states per chain 
% Mu - mean vectors
% Cov - output covariance matrix (full, tied across states)
% P - state transition matrix
% Pi - priors
%
% lik - log likelihood
%

function lik=calcmflike(X,T,mf,M,K,Mu,Cov,P,Pi);

p=length(X(1,:));
N=length(X(:,1));
if (rem(N,T)~=0)
  disp('Error: Data matrix length must be multiple of sequence length T');
  return;
end;
N=N/T;

k1=(2*pi)^(-p/2);

dd=zeros(K^M,M);
for i=1:K^M
  dd(i,:)=base(i-1,K,M);
end;

Mf=ones(T*N,K^M);
Mub=zeros(K^M,p);
for i=1:K^M
  dd(i,:)=base(i-1,K,M);
  for j=1:M;
    Mub(i,:)=Mub(i,:)+Mu((j-1)*K+dd(i,j),:);
    Mf(:,i)=Mf(:,i).*mf(:,(j-1)*K+dd(i,j));
  end;
end;

% to prevent log of zero:

logPi=log(Pi+(Pi==0)*exp(-744));
logP=log(P+(P==0)*exp(-744));
logmf=log(mf+(mf==0)*exp(-744));

lik=0;

iCov=inv(Cov);      
k2=k1/sqrt(det(Cov));
for l=1:(K^M)
  d= ones(N*T,1)*Mub(l,:)-X;
  lik=lik - 0.5*sum(Mf(:,l).*rsum((d*iCov).*d));
end; 

lik=lik+T*N*log(k2);

lik=lik+sum(mf(1:N,:)*logPi(:))-sum(sum(mf(1:N,:).*logmf(1:N,:)));


for i=2:T
  d1=(i-1)*N+1:i*N;
  d0=(i-2)*N+1:(i-1)*N;
  for j=1:M
    d2=(j-1)*K+1:j*K; 
    lik=lik+sum(sum(mf(d0,d2).*(mf(d1,d2)*logP(d2,:)')))-sum(sum(mf(d1,d2).*logmf(d1,d2)));
  end;
end;

