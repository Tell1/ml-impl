
function [Cp]=chorparse(C,p);

Cp=[];
Ctemp=[];
start=1;
for i=1:length(C);
  lin=C(i,:);
  if(sum(lin)==0) 
    start=i;
    Ctemp=[];
  end;
  if ((i-start-1)<p & i-start ~=0)
    Ctemp=[Ctemp; lin];
  elseif ((i-start-1)==p)
    % takes differences of first column
    C1=[0; Ctemp(:,1)];
    Ctemp(:,1)=diff(C1);
    Cp=[Cp;Ctemp];
  elseif ((i-start-1)>p);
  end;
end;
