% Character Sequence Demo
%
% demonstration of discrete HMM learning using character sequences
% from the book _Decline and Fall of the Roman Empire_ by Gibbon
% the characters are coded in a way described in charascii.m
% which can be used to look at the training data
%
% (loads a file 'chardat.mat' in the current directory
%  to get the character data)


load chardat


% 2-state HMM for character seqs
rand('state',108); % a good local optimum
[T2,p02,H2,ll2] = bw0x(dat0,2,[],[],1e-5,100);
  figure(1); clf; chardispH(H2(:,1),.7); set(gcf,'Position',[10,10,200,200]);
  figure(2); clf; chardispH(H2(:,2),.7); set(gcf,'Position',[220,10,200,200]);
  figure(3); clf; set(gcf,'Position',[430,10,200,200]);
  imagesc(T2); colormap gray; caxis([0 1]); axis xy; 
  set(gca,'XTick',1:size(T2,1)); set(gca,'YTick',1:size(T2,2));
  ylabel('from state'); xlabel('to state'); title('Transition Matrix');
  drawnow;

% 3-state HMM for character seqs
rand('state',1); % a good local optimum
[T3,p03,H3,ll3] = bw0x(dat0,3,[],[],1e-5,100);
  figure(4); clf; chardispH(H3(:,1),.7); set(gcf,'Position',[10,220,200,200]);
  figure(5); clf; chardispH(H3(:,2),.7); set(gcf,'Position',[220,220,200,200]);
  figure(6); clf; chardispH(H3(:,3),.7); set(gcf,'Position',[430,220,200,200]);
  figure(7); clf; set(gcf,'Position',[640,220,200,200]);
  imagesc(T3); colormap gray; caxis([0 1]); axis xy; 
  set(gca,'XTick',1:size(T3,1)); set(gca,'YTick',1:size(T3,2));
  ylabel('from state'); xlabel('to state'); title('Transition Matrix');
  drawnow;
