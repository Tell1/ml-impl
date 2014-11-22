function [T,p0,C,R,ll] = bw0(Y,T,p0,C,R,tol,maxiter,updateflags)
%[T,p0,C,R,ll] = bw0(Y,Ti,p0i,Ci,Ri,tol,maxiter,updateflags)
%
% TRAIN A HIDDEN MARKOV MODEL USING BAUM-WELCH 
%
% this routine does the initialization and controls how many
% iterations to train for
% see bw.m for the code that does one iteration
%
% Ti,p0i,Ri,Ci are initial estimates of the parameters
% if they are empty matrices [] then they are randomly initialized
% EXCEPT: if you want to initialize T randomly, make it a scalar
%         equal to the number of states
%
% tol is the fractional change in log likelihood to stop default 1e-5
% maxiters is the max# iterations of EM default 100
%
% updateflags controls the update of parameters
%   it is a four-vector whose elements control the updating of
%   [T,p0,C,R] -- nonzero elements mean update that parameter
%
%
% example   [T,p0,C,R,ll] = bw0(Y,10,[],[],R,1e-5,100,[1,1,1,0]);
% trains a 10 state HMM on data Y with covariance matrix fixed to R
% 

more off;
if(~iscell(Y)) Ytmp=cell(1,1); Ytmp{1}=Y; Y=Ytmp; clear Ytmp; end

pp = size(Y{1},1);

if(nargin<8) updateflags=[]; end
if(nargin<7) maxiter=[];     end
if(nargin<6) tol=[];         end
if(nargin<5) R=[];           end
if(nargin<4) C=[];           end
if(nargin<3) p0=[];          end

% initialize flags, iters, tol
if(isempty(updateflags)) updateflags=[1,1,1,1]; end
if(isempty(maxiter)) maxiter=100; end
if(isempty(tol)) tol=1e-5; end

% initialize T,p0,C,R
if(size(T,1)==1) kk=T; T=[]; else kk=size(T,1); end
if(isempty(T))  T = rand(kk,kk); T = T./(sum(T,2)*ones(1,kk));  end
                                 % make each row sum to unity
if(isempty(p0)) p0 = rand(kk,1); p0=p0/sum(p0);                 end
if(isempty(R))  R=diag(diag(cov(cat(2,Y{:})')));                end
if(isempty(C))  C=sqrtm(R)*randn(pp,kk)+mean(cat(2,Y{:}),2)*ones(1,kk); end
			 
ll=[-1e200,-1e199];

while(((ll(end)-ll(end-1))/abs(ll(end)) > tol) & (length(ll)<=maxiter+3))
  [T,p0,C,R,llnew] = bw(Y,T,p0,C,R,updateflags);
  ll = [ll,mean(llnew)];
  fprintf(1,'Iteration:\t%d\tlogLikelihood:%f',length(ll)-2,ll(end));
  if(length(ll)>3) fprintf(1,'\tdiff:%f',ll(end)-ll(end-1)); end
  fprintf(1,'\n');
end
ll=ll(3:end);

