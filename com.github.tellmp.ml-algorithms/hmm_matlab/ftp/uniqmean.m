% function Y=uniqmean(X);
% groups elements of X by first column of X and returns the mean of these
% groups in Y   

function Y=uniqmean(X);

idx=X(:,1);
N=length(idx);
tags=zeros(N,1);

Y=[];
for i=1:N;
  if (tags(i)==0)
    cidx=find(idx==idx(i));
    Li=length(cidx);
    if Li==1
      Y=[Y; X(cidx,:)];
    else
      Y=[Y; mean(X(cidx,:))];
    end;
    tags(cidx)=ones(Li,1);
  end;
end;