% Geyser data demo
%
% demonstration of continuous HMM learning using geyser data
% of intervals between erruptions and durations of erruptions
% of Old Faithful at Yellowstone National Park, USA
%
% (loads a file 'geyser.dat' in the current directory
%  to get the geyser data)


%uncomment the line below to see an initial setting that 
%takes a long time to converge to the global optimum
%rand('state',101);  randn('state',101);

%uncomment the line below to see an initial setting that converges to
%a bad local optimum
%rand('state',797);  randn('state',797);

more off;
Y = load('geyser.dat'); Y=Y';
[pp,tau] = size(Y);

% initialize flags, iters, tol
updateflags=[1,1,1,1];
maxiter=100;
tol=1e-6;

T=3; p0=[]; C=[]; R=[];	
[T,p0,C,R,ll] = bw0(Y,T,p0,C,R,0,1,updateflags); % just do one iteration

figure(1); set(gcf,'Position',[10,100,400,400]);
figure(2); set(gcf,'Position',[420,100,400,400]);

while(((ll(end)-ll(end-1)) > tol) & (length(ll)<=maxiter+3))
  
  % learning
  [T,p0,C,R,llnew] = bw(Y,T,p0,C,R,updateflags); % just do one iteration
  ll = [ll,llnew];
  
  % output progress as text
  if(length(ll)>3)
    fprintf(1,'Iteration:\t%d\tlogLikelihood:%f\tdiff:%f\n',...
        length(ll)-2,ll(end),ll(end)-ll(end-1));
  else
    fprintf(1,'Iteration:\t%d\tlogLikelihood:%f\n',...
        length(ll)-2,ll(end));
  end
  
  % plot progress
  figure(1)
  plot(Y(1,:),Y(2,:),'rx',C(1,:),C(2,:),'yo');
  hh = plotGauss(C(1,1),C(2,1),R(1,1),R(2,2),R(1,2));  
  set(hh,'LineWidth',2);
  hh = plotGauss(C(1,2),C(2,2),R(1,1),R(2,2),R(1,2));  
  set(hh,'LineWidth',2);
  hh = plotGauss(C(1,3),C(2,3),R(1,1),R(2,2),R(1,2));  
  set(hh,'LineWidth',2);
  xlabel('duration [min]');  
  ylabel('interval since last erruption [min]'); 
  hh=title('State output functions');
  set(hh,'FontSize',24);
  figure(2)
  imagesc(T); colormap gray; caxis([0 1]); axis xy; 
  set(gca,'XTick',1:size(T,1)); set(gca,'YTick',1:size(T,2));
  ylabel('from state'); xlabel('to state'); hh=title('Transition Matrix');
  set(hh,'FontSize',24);
  drawnow;

end

ll=ll(2:end);

