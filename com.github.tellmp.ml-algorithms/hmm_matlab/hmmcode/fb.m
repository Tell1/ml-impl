function [gamma,eta,rho] = fb(Y,T,p0,C,R,updateflags)
% [gamma,eta,rho] = fb(Y,T,p0,C,R,updataparams)
%
% FORWARD BACKWARD for state estimation in Hidden Markov Models
%
% Y is a matrix of observations, one per column
%
% T(i,j) is the probability of going to j next if you are now in i
% p0(j)  is the probability of starting in state j
% C(q,j) is the q^th coordinate of the j^th state mean vector
% R(q,r) is the covariance between the q and r coordinate for observations
%        or a scalar if R is a multiple of the identity matrix	
%
% updateflags controls the update of parameters
%   it is a four-vector whose elements control the updating of
%   [T,p0,C,R] -- nonzero elements mean update that parameter
%
% gamma(i,t) = p(x_t=i|Y) are the state inferences
% eta(i,j) = sum_t p(x_t=i,x_t+1=j|Y) are the transition counts
% rho(t) = p(y_t | y_1,y2,...y_t-1) are the scaling factors
%

if(nargin<6) updateflags=[1,1,1,1]; end

gamma=[]; eta=[]; rho=[];
if(any(updateflags))

% initial checking and nonsense
[pp,tau] = size(Y); [pp2,kk] = size(C); p0=p0(:);
assert(pp==pp2);  assert(length(p0)==kk);

% some constants
if(all(size(R)==1))
  intR=1; Rinv=1/R; z2=sqrt(Rinv^pp);
else
  intR=0; Rinv = inv(R); z2 = sqrt(det(Rinv));
end
z1 = (2*pi)^(-pp/2); zz=z1*z2;

% initialize space
alpha=zeros(kk,tau); beta=zeros(kk,tau); bb=zeros(kk,tau); rho=zeros(1,tau);

% compute bb
for ii=1:kk
  dd = Y-C(:,ii)*ones(1,tau);
  bb(ii,:) = zz*exp(-.5*sum((Rinv*dd).*dd,1));
end

% compute alpha, rho, beta
alpha(:,1) = p0.*bb(:,1);
rho(1)=sum(alpha(:,1));
alpha(:,1) = alpha(:,1)/rho(1);
for tt=2:tau
  alpha(:,tt) = (T'*alpha(:,tt-1)).*bb(:,tt);
  rho(tt) = sum(alpha(:,tt));
  alpha(:,tt) = alpha(:,tt)/rho(tt);
end
beta(:,tau) = 1;
for tt=(tau-1):-1:1
  beta(:,tt) = T*(beta(:,tt+1).*bb(:,tt+1))/rho(tt+1); 
end

% compute eta, AND sum it over all time (but don't normalize the result)
if(updateflags(1))
  eta = zeros(kk,kk);
  for tt=1:(tau-1)
    etatmp = T.*(alpha(:,tt)*(beta(:,tt+1).*bb(:,tt+1))');
    eta = eta+etatmp/rho(tt+1);
%    eta = eta+etatmp/sum(etatmp(:)); % same thing as above, just slower
  end
%  eta = eta./(sum(eta,2)*ones(1,kk)); % this would make each row sum to unity
                                       % but doesn't work with multiple seqs.
end

% compute gamma
if(any(updateflags(2:4)))
  gamma = (alpha.*beta);  % here we could just say alpha=alpha.*beta
                          % and avoid allocating the gamma memory
%  gamma = gamma./(ones(kk,1)*sum(gamma,1)); % only to avoid numerical noise
                                             % since gamma should be normalized
end


end
