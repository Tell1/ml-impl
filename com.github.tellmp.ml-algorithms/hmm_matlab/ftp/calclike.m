% function lik=calclike(X,T,M,K,Mu,Cov,P,Pi);
% 
% Calculate Likelihood for Factorial Hidden Markov Model 
% (Not using efficent Mth order decimation)
%
% X - N x p data matrix
% T - length of each sequence (N must evenly divide by T, default T=N)
% M - number of chains 
% K - number of states per chain 
% Mu - mean vectors
% Cov - output covariance matrix (full, tied across states)
% P - state transition matrix
% Pi - priors
%
% lik - log likelihood
%

function lik=calclike(X,T,M,K,Mu,Cov,P,Pi);

p=length(X(1,:));
N=length(X(:,1));
tiny=exp(-744);

if (rem(N,T)~=0)
  disp('Error: Data matrix length must be multiple of sequence length T');
  return;
end;
N=N/T;

alpha=zeros(T,K^M);
B=zeros(T,K^M);      % P( output | s_i) 
	
k1=(2*pi)^(-p/2);

dd=zeros(K^M,M);
for i=1:K^M
  dd(i,:)=base(i-1,K,M);
end;

Mub=zeros(K^M,p);
Pb=ones(K^M,K^M);
Pib=ones(K^M,1);
for i=1:K^M
  dd(i,:)=base(i-1,K,M);
  for j=1:M;
    Mub(i,:)=Mub(i,:)+Mu((j-1)*K+dd(i,j),:);
    Pib(i,:)=Pib(i,:)*Pi(dd(i,j),j);
  end;
  for j=1:K^M
    for l=1:M
      Pb(i,j)=Pb(i,j)*P((l-1)*K+dd(i,l),dd(j,l));
    end;
  end;
end;

Scale=zeros(T,1);

for n=1:N
  
  B=zeros(T,K^M); 
  iCov=inv(Cov);      
  k2=k1/sqrt(det(Cov));
  for i=1:T
    for l=1:(K^M)
      d= Mub(l,:)-X((n-1)*T+i,:);
      B(i,l)=k2*exp(-0.5*d*iCov*d');
    end; 
  end; 
  
  scale=zeros(T,1);
  alpha(1,:)=Pib(:)'.*B(1,:);
  scale(1)=sum(alpha(1,:)); 
  alpha(1,:)=alpha(1,:)/(scale(1)+tiny);
  for i=2:T
    alpha(i,:)=(alpha(i-1,:)*Pb).*B(i,:); 
    scale(i)=sum(alpha(i,:));
    alpha(i,:)=alpha(i,:)/(scale(i)+tiny);
  end;
  
  Scale=Scale+log(scale+tiny);
end;

lik=sum(Scale);
