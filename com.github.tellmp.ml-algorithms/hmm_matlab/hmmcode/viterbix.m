function [qq,ll] = viterbix(Y,T,p0,H)
% [qq,ll] = viterbix(Y,T,p0,H)
%
% VITERBI DECODING for state estimation in discrete Hidden Markov Models
%
% Y is a vector of integers (observed symbols)
%
% T(i,j) is the probability of going to j next if you are now in i
% p0(j)  is the probability of starting in state j
% H(m,j) is the probability of emitting symbol m if you are in state j
%
% qq is the state path most likely to have generated Y given the model params
% ll is the joint likelihood of Y and qq given the model params
%    divided by the length of Y (to be consistent with fb)


% initial checking and nonsense
tau = length(Y); [M,kk] = size(H); if(size(p0,2)~=1) p0=p0(:); end
assert(tau>0); assert(length(p0)==kk); assert(max(Y)<=M); assert(min(Y)>=1);

% initialize space
delta=zeros(kk,tau);  psi=zeros(kk,tau); qq=zeros(1,tau);

% compute bb
bb = H(Y,:)';

% take logs of parameters for numerical ease, then use addition
p0 = log(p0+eps); T = log(T+eps); bb = log(bb+eps);

delta(:,1) = p0+bb(:,1); 
psi(:,1)=0;

for tt=2:tau
  [delta(:,tt),psi(:,tt)] = max((delta(:,tt-1)*ones(1,kk)+T)',[],2);
  delta(:,tt) = delta(:,tt)+bb(:,tt);
end

[ll,qq(tau)] = max(delta(:,tau));
ll=ll/tau;

for tt=(tau-1):-1:1
  qq(tt)=psi(qq(tt+1),tt+1);
end

