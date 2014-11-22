% function [Mu,Cov,P,Pi,LL,TL]=pmhmm(X,T,M,K,cyc,tol,iter);
% 
% Partial Mean Field Factorial Hidden Markov Model 
%
% X - N x p data matrix
% T - length of each sequence (N must evenly divide by T, default T=N)
% M - number of chains (default 2)
% K - number of states per chain (default 2)
% cyc - maximum number of cycles of Baum-Welch (default 100)
% tol - termination tolerance (prop change in likelihood) (default 0.0001)
% iter - maximum number of MF iterations (default 10)
%
% Mu - mean vectors
% Cov - output covariance matrix (full, tied across states)
% P - state transition matrix
% Pi - priors
% LL - log likelihood curve
% TL - true log likelihood curve (optional -- slows computation)
%
% Iterates until a proportional change < tol in the log likelihood 
% or cyc steps of Baum-Welch

% This version has automatic MF stopping 

function [Mu,Cov,P,Pi,LL,TL]=pmhmm(X,T,M,K,cyc,tol,iter)

p=length(X(1,:));
N=length(X(:,1));
tiny=exp(-700);

if nargin<7   iter=10; end;
if nargin<6   tol=0.0001; end;
if nargin<5   cyc=100; end;
if nargin<4   K=2; end;
if nargin<3   M=2; end;
if nargin<2   T=N; end;

if (rem(N,T)~=0)
  disp('Error: Data matrix length must be multiple of sequence length T');
  return;
end;
N=N/T;

Cov=diag(diag(cov(X))); 
XX=X'*X/(N*T);

% X reordered with TN*p rather than NT*p :
Xalt=[];
ai=1:N*T;
for i=1:T
  indxi=rem(ai-i,T)==0;
  Xalt=[Xalt; X(indxi,:)];
end;

Mu=randn(M*K,p)*sqrtm(Cov)/M+ones(K*M,1)*mean(X)/M;

Pi=rand(K,M);
Pi=cdiv(Pi,csum(Pi));

P=rand(K*M,K);
P=rdiv(P,rsum(P));

LL=[];
lik=0;
TL=[];

gamma=zeros(N*T,K*M);    % P ( s_i | O) Gamma is (T*N,K*M)
k1=(2*pi)^(-p/2);

mf=ones(N*T,M*K)/K; 
h=ones(N*T,M*K)/K;
alpha=zeros(N*T,M*K);
beta=zeros(N*T,M*K);
logmf=log(mf);
exph=exp(h);

for cycle=1:cyc
  
  %%%% FORWARD-BACKWARD %%% MF E STEP
  
  Gamma=[];
  GammaX=zeros(M*K,p);
  Eta=zeros(K*M,K*M); 
  Xi=zeros(M*K,K);

  % Solve mean field equations for all input sequences

  iCov=inv(Cov);      
  k2=k1/sqrt(det(Cov));
  
  itermf=iter;
  for l=1:iter
    
    mfo=mf;
    logmfo=logmf;

    %%%% first compute h values based on mf
    
    for i=1:T
      d2=(i-1)*N+1:i*N;
      d = Xalt(d2,:); % N*p
      % compute yhat (N*p);
      yhat=mf(d2,:)*Mu;
      for j=1:M
	d1=(j-1)*K+1:j*K;
	Muj=Mu(d1,:); % Kxp
	mfold=mf(d2,d1);
	logP=log(P(d1,:)+tiny);
	logPi=log(Pi(:,j)+tiny);
	h(d2,d1)=(Muj*iCov*(d-yhat)'+Muj*iCov*Muj'*mf(d2,d1)' ...
	    - 0.5*diag(Muj*iCov*Muj')*ones(1,N))';
	h(d2,d1)=h(d2,d1)-max(h(d2,d1)')'*ones(1,K);
      end; %j
    end; %i
    exph=exp(h);
    
    %%% then compute mf values based on h using Baum-Welch
    
    scale=zeros(T*N,M);
    for j=1:M
      d1=(j-1)*K+1:j*K;
      d2=1:N;
      alpha(d2,d1)=exph(d2,d1).*(ones(N,1)*Pi(:,j)');
      scale(d2,j)=rsum(alpha(d2,d1))+tiny;
      alpha(d2,d1)=rdiv(alpha(d2,d1),scale(d2,j));
      for i=2:T
	d2=(i-1)*N+1:i*N;
	alpha(d2,d1)=(alpha(d2-N,d1)*P(d1,:)).*exph(d2,d1);
	scale(d2,j)=rsum(alpha(d2,d1))+tiny;
	alpha(d2,d1)=rdiv(alpha(d2,d1),scale(d2,j));
      end;
      
      d2=(T-1)*N+1:T*N;
      beta(d2,d1)=rdiv(ones(N,K),scale(d2,j));
      for i=T-1:-1:1
	d2=(i-1)*N+1:i*N;
	beta(d2,d1)=(beta(d2+N,d1).*exph(d2+N,d1))*(P(d1,:)');
	beta(d2,d1)=rdiv(beta(d2,d1),scale(d2,j));
      end;
    
      mf(:,d1)=(alpha(:,d1).*beta(:,d1));
      mf(:,d1)=rdiv(mf(:,d1),rsum(mf(:,d1))+tiny);
    end;    
    
    logmf=log(mf+(mf==0).*tiny);
    delmf=sum(sum(mf.*logmf))-sum(sum(mf.*logmfo));
    if(delmf<N*T*1e-6) 
      itermf=l;
      break;
    end;
  end; %iter

  % calculating mean field log likelihood 
    
  if (nargout>=5)
    oldlik=lik;
    lik=calcmflike(Xalt,T,mf,M,K,Mu,Cov,P,Pi);
  end;
  
  %  first and second order correlations - P (s_i, s_j | O)

  gamma=mf;
  Gamma=gamma;
  Eta=gamma'*gamma;
  gammasum=sum(gamma);
  for j=1:M
    d2=(j-1)*K+1:j*K;
    Eta(d2,d2)=diag(gammasum(d2)); 
  end;
  
  GammaX=gamma'*Xalt;
  Eta=(Eta+Eta')/2;

  Xi=zeros(M*K,K); 
  for i=1:T-1
    d1=(i-1)*N+1:i*N;
    d2=d1+N;
    for j=1:M
      jK=(j-1)*K+1:j*K;
      % t=gamma(d1,jK)'*gamma(d2,jK);
      t = P(jK,:).*(alpha(d1,jK)'*(beta(d2,jK).*exph(d2,jK)));
      Xi(jK,:)=Xi(jK,:)+t/sum(t(:));
    end;
  end;
  
  %%%% Calculate Likelihood and determine convergence
  
  LL=[LL lik];
  if (nargout>=6)
    truelik=calclike(X,T,M,K,Mu,Cov,P,Pi);
    TL=[TL truelik];
    fprintf('cycle %i mf iters %i log like= %f true log like= %f',cycle,itermf,lik,truelik);  
  elseif (nargout==5)
    fprintf('cycle %i mf iters %i log likelihood = %f ',cycle,itermf,lik);
  else
    fprintf('cycle %i mf iters %i ',cycle,itermf);
  end;
  
  if (nargout>=5)
    if (cycle<=2)
      likbase=lik;
    elseif (lik<oldlik-2)
      fprintf('v');
    elseif (lik<oldlik) 
      fprintf('v');
    elseif ((lik-likbase)<(1 + tol)*(oldlik-likbase)) 
      fprintf('\n');
      break;
    end;
  end;
  fprintf('\n');

  %%%% M STEP 
  
  % outputs -- using SVD as generally ill-conditioned (Mu=pinv(Eta)*GammaX);
  [U,S,V]=svd(Eta);
  Si=zeros(K*M,K*M);
  for i=1:K*M;
    if(S(i,i)<max(size(S))*norm(S)*0.001)
      Si(i,i)=0;
    else Si(i,i)=1/S(i,i);
    end;
  end;  
  Mu=V*Si*U'*GammaX;
  
  % covariance
  Cov=XX-GammaX'*pinv(Eta)*GammaX/(N*T);
  Cov=(Cov+Cov')/2;
  dCov=det(Cov);
  while (dCov<0)
    fprintf('\nAbort: covariance problem \n');     return;
  end;
  
  % transition matrix 
  for i=1:K*M
    d1=sum(Xi(i,:));
    if(d1==0)
      P(i,:)=ones(1,K)/K;
    else
      P(i,:)=Xi(i,:)/d1;
    end;
  end;
  
  % priors (note Gamma in NT order not TN order)
  
  Pi=reshape(csum(Gamma(1:N,:)),K,M)/N;

end
