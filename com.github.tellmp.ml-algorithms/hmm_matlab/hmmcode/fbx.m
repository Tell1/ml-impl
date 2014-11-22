function [gamma,eta,rho] = fbx(Y,T,p0,H,updateflags)
% [gamma,eta,rho] = fbx(Y,T,p0,H,updateflags)
%
% Forward Backward for DISCRETE symbol HMMs
%
% Y is a vector of integers (observed symbols)
%
% T(i,j) is the probability of going to j next if you are now in i
% p0(j)  is the probability of starting in state j
% H(m,j) is the probability of emitting symbol m if you are in state j
%
% updateflags controls the update of parameters
%   it is a three-vector whose elements control the updating of
%   [T,p0,H] -- nonzero elements mean update that parameter
%
% gamma(i,t) = p(x_t=i|Y) are the state inferences
% eta(i,j) = sum_t p(x_t=i,x_t+1=j|Y) are the transition counts
% rho(t) = p(y_t | y_1,y2,...y_t-1) are the scaling factors
%

if(nargin<5) updateflags=[1,1,1]; end

gamma=[]; eta=[]; rho=[];
if(any(updateflags))

% initial checking and nonsense
tau = length(Y); [M,kk] = size(H); if(size(p0,2)~=1) p0=p0(:); end
assert(tau>0); assert(length(p0)==kk); assert(max(Y)<=M); assert(min(Y)>=1);

% initialize space
alpha=zeros(kk,tau); beta=zeros(kk,tau); rho=zeros(1,tau);

% compute bb
bb = H(Y,:)';

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
if(any(updateflags(2:3)))
  gamma = (alpha.*beta);  % here we could just say alpha=alpha.*beta
                          % and avoid allocating the gamma memory
%  gamma = gamma./(ones(kk,1)*sum(gamma,1)); % only to avoid numerical noise
                                             % since gamma should be normalized
end


end
