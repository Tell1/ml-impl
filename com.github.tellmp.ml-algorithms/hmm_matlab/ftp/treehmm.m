% function [Mu,Cov,P1,P2,P3,Pi1,Pi2,Pi3,LL]=treehmm(X,T,M,K,cyc,tol,iter)
% 
% Structured Mean Field for Decision Tree Hidden Markov Model 
%
% X - N x p data matrix
% T - length of each sequence (N must evenly divide by T, default T=N)
% M - levels in tree (default 2)
% K - number of states per node (branching factor) (default 2)
% cyc - maximum number of cycles of Baum-Welch (default 100)
% tol - termination tolerance (prop change in likelihood) (default 0.0001)
% iter - maximum number of MF iterations (default 10)
%
% Mu - mean vectors
% Cov - output covariance matrix (full, tied across states)
% P1, P2, P3 - state transition matrices
% Pi1, Pi2, Pi3 - priors
% LL - log likelihood curve
%
% Iterates until a proportional change < tol in the log likelihood 
% or cyc steps of Baum-Welch

% This version has automatic MF stopping 

function [Mu,Cov,P1,P2,P3,Pi1,Pi2,Pi3,LL]=treehmm(X,T,M,K,cyc,tol,iter)

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

Xalt=zeros(N*T,p);
if (N~=1 & T~=1)
  ai=1:N*T;
  for i=1:T
    indxi=find(rem(ai-i,T)==0);
    Xalt((i-1)*N+1:i*N,:)=X(indxi,:);
  end;
else
  Xalt=X; 
end;

Mu=0.2*randn(M*K,p)*sqrtm(Cov)/M+ones(K*M,1)*mean(X)/M

Pi1=rnorm(rand(1,K)+10);
Pi2=rnorm(rand(K,K)+10);
Pi3=rnorm(rand(K^2,K)+10);

% P1=rnorm(rand(K,K)+10);
% P2=rnorm(rand(K^2,K)+10);
% P3=rnorm(rand(K^3,K)+10);

P1=rnorm(eye(K)+1/2);
P2=rnorm(kron((eye(K)+2),ones(K,1)));
P3=rnorm(kron((eye(K)+4),ones(K^2,1)));


LL=[];
lik=0;

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
  for it=1:iter
    
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
	h(d2,d1)=(Muj*iCov*(d-yhat)'+Muj*iCov*Muj'*mf(d2,d1)' ...
	    - 0.5*diag(Muj*iCov*Muj')*ones(1,N))';
	% h(d2,d1)=h(d2,d1)-log(rsum(exp(h(d2,d1))))*ones(1,K);
      end; %j
    end; %i
    exph=exp(h);
    
    %%% then compute mf values based on h using Baum-Welch

    scale=zeros(T*N,M);
    for j=1:M
      d1=(j-1)*K+1:j*K;
      B=zeros(N*T*K,K);
      d2=1:N;
      if (j==1)
	alpha(d2,d1)=exph(d2,d1).*(ones(N,1)*Pi1);
      elseif (j==2)
	alpha(d2,d1)=exph(d2,d1).*(mf(d2,d1-K)*Pi2);
      elseif (j==3)
	for l=1:N
	  alpha(d2(l),d1)=exph(d2(l),d1).*(...
	      kron(mf(d2(l),d1-K),mf(d2(l),d1-2*K))*Pi3);
	end;
	% keyboard;
      end;
      scale(d2,j)=rsum(alpha(d2,d1))+tiny;
      alpha(d2,d1)=rdiv(alpha(d2,d1),scale(d2,j));
      for i=2:T
	d2=(i-1)*N+1:i*N;
	
	if (j==1)
	  alpha(d2,d1)=(alpha(d2-N,d1)*P1).*exph(d2,d1);
	elseif (j==2)
	  for l=1:N
	    d2l=d2(l);
	    mft=kron(eye(K),mf(d2l,d1-K));
	    P=rnorm(exp(mft*log(P2+tiny)));
	    alpha(d2l,d1)=(alpha(d2l-N,d1)*P).*exph(d2(l),d1);
	    B(((d2l-1)*K+1:d2l*K)-K*N,:)=P;
	  end;
	elseif (j==3)
	  for l=1:N
	    d2l=d2(l);
	    mft=kron(eye(K),kron(mf(d2l,d1-K),mf(d2l,d1-2*K)));
	    P=rnorm(exp(mft*log(P3+tiny)));
	    alpha(d2l,d1)=(alpha(d2l-N,d1)*P).*exph(d2l,d1);
	    B(((d2l-1)*K+1:d2l*K)-K*N,:)=P;
	  end;
	end;
	scale(d2,j)=rsum(alpha(d2,d1))+tiny;
	alpha(d2,d1)=rdiv(alpha(d2,d1),scale(d2,j));
      end;
      
      d2=(T-1)*N+1:T*N;
      beta(d2,d1)=rdiv(ones(N,K),scale(d2,j));

      for i=T-1:-1:1
	d2=(i-1)*N+1:i*N;

	if (j==1)
	  beta(d2,d1)=(beta(d2+N,d1).*exph(d2+N,d1))*(P1');
	else
	  for l=1:N
	    d2l=d2(l);
	    beta(d2l,d1)=(beta(d2l+N,d1).*exph(d2l+N,d1))*...
		(B((d2l-1)*K+1:d2l*K,:)'); 
	  end;
	end;
	  
	beta(d2,d1)=rdiv(beta(d2,d1),scale(d2,j));
      end;
      mf(:,d1)=(alpha(:,d1).*beta(:,d1));
      mf(:,d1)=rnorm(mf(:,d1));
    end;     
   
    logmf=log(mf+(mf==0).*tiny);
    delmf=sum(sum(mf.*logmf))-sum(sum(mf.*logmfo));
    if(delmf<N*T*1e-6) 
      itermf=it;
      break;
    end;
  end; %iter

  % calculating mean field log likelihood 
    
  if (nargout>=9)
    oldlik=lik;
    % lik=treehmm_cl(Xalt,T,mf,M,K,Mu,Cov,P1,P2,P3,Pi1,Pi2,Pi3);
    lik=sum(sum(log(scale)));
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

  Xi1=zeros(K,K); 
  Xi2=zeros(K^2,K); 
  Xi3=zeros(K^3,K); 

  for i=1:T-1
    d1=(i-1)*N+1:i*N;
    d2=d1+N;
    for j=1:M
      jK=(j-1)*K+1:j*K;
      if (j==1)
	t = P1.*(alpha(d1,jK)'*(beta(d2,jK).*exph(d2,jK)));
	Xi1=Xi1+t/sum(t(:));
      elseif (j==2)
	alphat=zeros(N,K^2);
	for l=1:N
	  alphat(l,:)=kron(alpha(d1(l),jK),mf(d1(l),jK-K));
	end;
	t = P2.*(alphat'*(beta(d2,jK).*exph(d2,jK)));
	Xi2=Xi2+t/sum(t(:));
      elseif (j==3)
	alphat=zeros(N,K^3);
	for l=1:N
	  alphat(l,:)=kron(alpha(d1(l),jK),kron(mf(d1(l),jK-K),...
	      mf(d1(l),jK-2*K)));
	end;
	t = P3.*(alphat'*(beta(d2,jK).*exph(d2,jK)));
	Xi3=Xi3+t/sum(t(:));
      end;
    end;
  end;
  
  %%%% Calculate Likelihood and determine convergence
  
  LL=[LL lik];
  if (nargout>=9)
    fprintf('cycle %i mf iters %i log likelihood = %f ',cycle,itermf,lik);
  else
    fprintf('cycle %i mf iters %i ',cycle,itermf);
  end;
  
  if (nargout>=9)
    if (cycle<=2)
      likbase=lik;
    elseif (lik<oldlik) 
      fprintf('v');
      % keyboard;
    elseif ((lik-likbase)<(1 + tol)*(oldlik-likbase)|~finite(lik)) 
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
  Mu'
  
  % covariance
  Cov=XX-GammaX'*pinv(Eta)*GammaX/(N*T);
  Cov=(Cov+Cov')/2;
  dCov=det(Cov);
  while (dCov<0)
    fprintf('\nAbort: covariance problem \n');     return;
  end;
  
  % transition matrices
  P1=rnorm(Xi1);
  P2=rnorm(Xi2);
  P3=rnorm(Xi3);
  
  % priors 
  
  Pi1=csum(mf(1:N,1:K))/N;
  Pi2=mf(1:N,1:K)'*mf(1:N,K+1:2*K);
  Pi2=rnorm(Pi2);
  Pi3=zeros(K^2,K);
  for l=1:N
    Pi3=Pi3+kron(mf(l,1:K),mf(l,K+1:2*K))'*mf(l,2*K+1:3*K);
  end;
  Pi3=rnorm(Pi3);

  P1
  P2
  P3
  
end
