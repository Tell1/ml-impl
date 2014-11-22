function dispH(hh,fs)
% dispH(hh,fontsize)
% displays a histogram vector hh showing graphically the letters
% that it places high probability on
%
% the font size of the letter is proportional to the square root
% of the prob, making the area roughly and indicator of the prob.
%

if(nargin<2) fs=1; end

iSPC=1;
iPUNC=2;
iDOT=3;
iNUM=4;
iALPHA=5;
iMAX=30;

cla;
axis([.5 6.5 .5 5.5]);
set(gca,'XTick',[]); set(gca,'YTick',[]);

tt=text(1,5,'-'); set(tt,'FontSize',fntsize(hh(iSPC),fs));
tt=text(1,4,'*'); set(tt,'FontSize',fntsize(hh(iDOT),fs));
tt=text(1,3,'#'); set(tt,'FontSize',fntsize(hh(iPUNC),fs));
tt=text(1,2,'9'); set(tt,'FontSize',fntsize(hh(iNUM),fs));

for ii=2:6;for jj=1:5;
qq=(ii-2)*5+jj-1;
tt=text(jj+1,7-ii,char(qq+65));
set(tt,'FontSize',fntsize(hh(iALPHA+qq),fs));
end;end;

qq=26-1;
tt=text(1,1,char(qq+65));
set(tt,'FontSize',fntsize(hh(iALPHA+qq),fs));


axis square



function [fsout] = fntsize(val,fs)

%fs=get(gca,'FontSize');
fsout=fs*150*sqrt(val);
fsout=max(fsout,2);
fsout=min(fsout,128);


