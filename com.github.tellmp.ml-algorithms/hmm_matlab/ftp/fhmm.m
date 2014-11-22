% function [Mu,Cov,P,Pi,LL]=fhmm(X,T,M,K,cyc,tol);
% 
% Factorial Hidden Markov Model with Gaussian Observations
%
% X - N x p data matrix
% T - length of each sequence (N must evenly divide by T, default T=N)
% M - number of chains (default 2)
% K - number of states per chain (default 2)
% cyc - maximum number of cycles of Baum-Welch (default 30)
% tol - termination tolerance (prop change in likelihood) (default 0.0001)
%
% Mu - mean vectors
% Cov - output covariance matrix (full, tied across states)
% P - state transition matrix
% Pi - priors
% LL - log likelihood curve
%
% Iterates until a proportional change < tol in the log likelihood 
% or cyc steps of Baum-Welch

function [Mu,Cov,P,Pi,LL]=fhmm(X,T,M,K,cyc,tol)

p=length(X(1,:));
N=length(X(:,1));

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

Mu=randn(M*K,p)*sqrtm(Cov)/M+ones(K*M,1)*mean(X)/M;

Pi=rand(K,M);
Pi=cdiv(Pi,csum(Pi)); 

P=rand(K*M,K);
P=rdiv(P,rsum(P));

LL=[];
lik=0;

dd=zeros(K^M,M);
for i=1:K^M
  dd(i,:)=base(i-1,K,M);
end;

alpha=zeros(T,K^M);
B=zeros(T,K^M);        % P( output | s_i) 
beta=zeros(T,K^M);
gamma=zeros(T,K^M);    % gamma1 is (T,K*M) Gamma is (T*N,K*M)
B2=zeros(T*K^M,K^M);   % P( output | s_i, s_j)

eta =zeros(T*K*M,K*M); % P (s_i, s_j | O)

Mub=zeros(K^M,p);
Pb=ones(K^M,K^M);
Pib=ones(K^M,1);
collapse=zeros(K^M,M*K);
collapse2=zeros(K^M,M*K*M*K); % huge matrix
for i=1:K^M
  dd(i,:)=base(i-1,K,M);
  for j=1:M;
    Mub(i,:)=Mub(i,:)+Mu((j-1)*K+dd(i,j),:);
    Pib(i,:)=Pib(i,:)*Pi(dd(i,j),j);
    collapse(i,(j-1)*K+dd(i,j))=1;
    for l=1:M
      collapse2(i,((j-1)*K+dd(i,j)-1)*M*K+(l-1)*K+dd(i,l))=1;
    end;
  end;
  for j=1:K^M
    for l=1:M
      Pb(i,j)=Pb(i,j)*P((l-1)*K+dd(i,l),dd(j,l));
    end;
  end;
end;

k1=(2*pi)^(-p/2);

for cycle=1:cyc
  
  %%%% FORWARD-BACKWARD %%% EXACT E STEP
  
  Gamma=[];
  GammaX=zeros(K*M,p);
  Eta=zeros(K*M,K*M); 
  Scale=zeros(T,1);
  Xi=zeros(K*M,K);
  
  % expand
  Mub=zeros(K^M,p);
  Pb=ones(K^M,K^M);
  Pib=ones(K^M,1);
  for i=1:K^M
    for l=1:M
      Mub(i,:)=Mub(i,:)+Mu((l-1)*K+dd(i,l),:);
      Pib(i,:)=Pib(i,:)*Pi(dd(i,l),l);
    end;
    for j=1:K^M
      for l=1:M
	Pb(i,j)=Pb(i,j)*P((l-1)*K+dd(i,l),dd(j,l));
      end;
    end;
  end;
    
  for n=1:N
   
    B=zeros(T,K^M); 
    iCov=inv(Cov);      
    k2=k1/sqrt(det(Cov));
    for l=1:(K^M),
      d=ones(T,1)*Mub(l,:)-X((n-1)*T+1:n*T,:);
      B(:,l)=k2*exp(-0.5*rsum((d*iCov).*d));
    end;
    
    scale=zeros(T,1);
    alpha(1,:)=Pib'.*B(1,:);
    scale(1)=sum(alpha(1,:)); 
    alpha(1,:)=alpha(1,:)/scale(1);
    for i=2:T
      alpha(i,:)=(alpha(i-1,:)*Pb).*B(i,:); 
      scale(i)=sum(alpha(i,:));
      alpha(i,:)=alpha(i,:)/scale(i);
    end;

    beta(T,:)=ones(1,K^M)/scale(T);
    for i=T-1:-1:1
      beta(i,:)=(beta(i+1,:).*B(i+1,:))*(Pb')/scale(i); 
    end;
    
    gamma=(alpha.*beta); 
    gamma=rdiv(gamma,rsum(gamma));
    
    gamma1=gamma*collapse;
    for i=1:T
      for j=1:M
	d1=(j-1)*K+1:j*K;
	gamma1(i,d1)=gamma1(i,d1)/sum(gamma1(i,d1));
      end;
    end;
	
    xi=zeros(M*K,K);
    for i=1:T-1
      t=(alpha(i,:)*collapse)'*((beta(i+1,:).*B(i+1,:))*collapse);
      for j=1:M
	d1=(j-1)*K+1:j*K;
	t2=P(d1,:).*t(d1,d1);
	xi(d1,:)=xi(d1,:)+t2/sum(t2(:));
      end;    
    end;
    
    t=gamma*collapse2;
    
    for i=1:T
      d1=(i-1)*K*M+1:i*K*M;
      eta(d1,:)=reshape(t(i,:),K*M,K*M);
      for j=1:M
	d2=(i-1)*K*M+(j-1)*K+1:(i-1)*K*M+j*K;
	for l=1:M
	  if (j==l)
	    eta(d2,(j-1)*K+1:j*K)=diag(gamma1(i,(j-1)*K+1:j*K)); 
	  else
	    d3=sum(sum(eta(d2,(l-1)*K+1:l*K)));
	    eta(d2,(l-1)*K+1:l*K)=eta(d2,(l-1)*K+1:l*K)/d3;
	  end;
	end;
      end;
      Eta=Eta+eta(d1,:);
      GammaX=GammaX+gamma1(i,:)'*X((n-1)*T+i,:);
    end;
  
    Scale=Scale+log(scale);
    Gamma=[Gamma; gamma1];
    Xi=Xi+xi;
  end;
  Eta=(Eta+Eta')/2;
  
  %%%% Calculate Likelihood and determine convergence
  
  oldlik=lik;
  lik=sum(Scale);
  LL=[LL lik];
  fprintf('cycle %i log likelihood = %f ',cycle,lik);
  
  if (cycle<=2)
    likbase=lik;
  elseif (lik<oldlik-2) 
    fprintf('Large likelihood violation \n');    
  elseif (lik<oldlik) 
    fprintf('v');
  elseif ((lik-likbase)<(1 + tol)*(oldlik-likbase)|~finite(lik)) 
    fprintf('\n');
    break;
  end;
  
  %%%% M STEP 
  
  % outputs -- using SVD as generally ill-conditioned (Mu=pinv(Eta)*GammaX);
  
  Mu=pinv(Eta)*GammaX;
  
  % covariance
  Cov=XX-GammaX'*Mu/(N*T);
  
  Cov=(Cov+Cov')/2;
  dCov=det(Cov);
  while (dCov<=0)
    fprintf('Covariance problem');
    Cov=Cov+eye(p)*(1.5*(-dCov)^(1/p)+eps);
    dCov=det(Cov);
  end;
  fprintf('\n');
  
  % transition matrix 
  for i=1:K*M
    d1=sum(Xi(i,:));
    if(d1==0)
      P(i,:)=ones(1,K)/K;
    else
      P(i,:)=Xi(i,:)/d1;
    end;
  end;
  
  % priors
  Pi=zeros(K,M);
  for i=1:N
    Pi=Pi+reshape(Gamma((i-1)*T+1,:),K,M);
  end
  Pi=Pi/N;
  
  % fprintf('computing M step: %i\n\n',flops);     flops(0);
  
end







