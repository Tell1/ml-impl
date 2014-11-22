function [cc] = charascii(dat)
% [cc] = charascii(dat)
%
% Converts numeric sequence dat into character sequence cc
% readable by humans.

iSPC=1;
iPUNC=2;
iDOT=3;
iNUM=4;
iALPHA=5;
iMAX=30;

dd=dat;

cc(find(dd==iSPC ))=32;
cc(find(dd==iPUNC))=35;
cc(find(dd==iDOT ))=46;

cc(find(dd==iNUM))=57;

ff=find((dd>=iALPHA) & (dd<=iMAX));
cc(ff)=65+dd(ff)-iALPHA;

cc=char(cc);
