function [Tnew,p0new,Cnew,Rnew,ll] = bw(Y,T,p0,C,R,updateflags)
% [Tnew,p0new,Cnew,Rnew,ll] = bw(Y,T,p0,C,R,updateflags)
%
% BAUM-WLECH updating for HMMs
% do one iteration of BaumWelch to update HMM params
% calls fb.m to do the forward-backward E-step
%
% Y is a matrix of observations, one per column
%   (or cell array of matrices if many sequences)
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
% ll is a scalar (or vector for multiple sequences) that holds 
%    the log likelihood per symbol (ie total divided by seq length)
%

if(nargin<6) updateflags=[1,1,1,1]; end
assert(length(updateflags==4));

ll=[]; tautot=0;
if(~iscell(Y)) Ytmp=cell(1,1); Ytmp{1}=Y; Y=Ytmp; clear Ytmp; end

pp = size(Y{1},1);
[pp2,kk] = size(C);
assert(pp==pp2);

if(updateflags(1)) Tnew  = zeros(size(T));   else Tnew =T;   end  
if(updateflags(2)) p0new = zeros(size(p0));  else p0new=p0; end
if(updateflags(3)) Cnew  = zeros(size(C));   else Cnew =C;   end
if(updateflags(4)) Rnew  = zeros(size(R));   else Rnew =R;   end

if(any(updateflags))
  ll=zeros(size(Y));
  gammasum=zeros(1,kk);
  for seqs=1:length(Y)    
    tau = length(Y{seqs}); tautot=tautot+tau;
    [gamma,eta,rho] = fb(Y{seqs},T,p0,C,R,updateflags); 
    if(updateflags(1)) Tnew  = Tnew  + eta;         end
    if(updateflags(2)) p0new = p0new + gamma(:,1);  end
    if(updateflags(3)) 
      Cnew = Cnew + (Y{seqs}*gamma');
      gammasum = gammasum+sum(gamma,2)';
    end
    if(updateflags(4))
      for jj=1:kk
        dd = Y{seqs}-C(:,jj)*ones(1,tau);
        Rnew = Rnew+(dd.*(ones(pp,1)*gamma(jj,:)))*dd';
      end
    end
    ll(seqs)=sum(log(rho))/tau;
  end
  % normalize probability distributions
  if(updateflags(1)) Tnew  = Tnew./(sum(Tnew,2)*ones(1,kk)); end
  if(updateflags(2)) p0new = p0new/sum(p0new);               end
  if(updateflags(3)) Cnew = Cnew./(ones(pp,1)*gammasum);     end
  if(updateflags(4)) Rnew=Rnew/tautot;                       end
end

