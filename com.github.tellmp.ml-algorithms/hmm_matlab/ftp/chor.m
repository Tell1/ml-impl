% run from chorales directory

load -ascii chorales.num
C=chorales;
Starts=find(rsum(chorales)==0); % find the starting indices
D=diff(C(:,1));
D=[0;D];
C(:,1)=(C(:,1)~=0).*D;
N=length(C);
C=cdiv((C-ones(N,1)*mean(C)),std(C));
plot(C+4*ones(N,1)*(6:-1:1));

Lns=diff(Starts);
hist(Lns,20); % histgram of chorale lengths

T=40;
Cp=chorparse(chorales,T);
length(Cp); % 2640 (i.e. 66 chorales of length 40)
Cp=Cp+randn(size(Cp))*0.1;
Cptrain=Cp(1:30*T,:);
Cpval=Cp((30*T+1):48*T,:);
Cptest=Cp((48*T+1):66*T,:);

% fit various HMMs and look at validation error

diary chdiary1;

liks=[];
for K=[2:10 12:2:20 25 30 ]; % this loop should take about 2 hours
  for rep=1:3
    fprintf('K = %g\n',K);
    [Mu,Cov,P,Pi,LL]=hmm(Cptrain,T,K);
    cyc=length(LL);
    trlik=LL(cyc);
    valik=hmm_cl(Cpval,T,K,Mu,Cov,P,Pi);
    telik=hmm_cl(Cptest,T,K,Mu,Cov,P,Pi);
    liks=[liks; K cyc trlik valik telik];
  end;
  liks
end;

save chworld1;
diary off;

a=plot(liks(:,1),liks(:,4)/720,'go','LineWidth',2);
hold on;
avliks=uniqmean(liks);
a=plot(avliks(:,1),avliks(:,4)/720,'-','LineWidth',2,'Color',[0 0.6 0]);
axis([0 80 -10 -4]);
xlabel('Size of state space','FontSize',18);
ylabel('Validation set log likelihood','FontSize',18);
title('HMM model of Bach Chorales','FontSize',20);
set(gca,'FontSize',18);

% plot(avliks(:,1),avliks(:,3)/1200,'b-');
% plot(avliks(:,1),avliks(:,5)/720,'r-');
