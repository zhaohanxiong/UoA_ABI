function MoeAFModel
%
% https://au.mathworks.com/matlabcentral/fileexchange/19855-model-of-atrial-fibrillation-flutter?s_tid=srchtitle
%
% AFIB   Computer model of atrial fibrillation model by Moe et al, 
%        American Heart Journal 1964; 67(2):200-220.
%
% Implemented in Matlab by Peter Hammer, Children's Hospital Boston, Feb 2002.
%
%       Cellular automata model simulating atrial flutter/fibrillation.
%       A cluster of (4) cells is rapidly stimulated causing periodic waves
%       of activation to spread across the model surface. Due to variation
%       in refractory periods across the surface, the wavefronts are broken
%       up as they collide with regions that are still refractory. The
%       rapid stimulation is terminated after 80 steps yet activation
%       wavefronts still travel throughout the model surface. The authors
%       carefully chose the cluster of cells to stimulate by looking for
%       adjacent cells with short refractory periods. I do not do that
%       here; rather I just keep varying the initial state of the random
%       number generator used to asign the K value until I get
%       self-sustaining activity. 
%
%       In the figure, three discrete states are plotted: activated or scheduled (blue), 
%       refractory (light blue), and resting/excitable (white). The time step
%       number of the simulation is displayed below the activation pattern.
%       To modulate the speed of the simulation, change the input argument
%       to the pause statement (time, in seconds, to pause during each step).
%
%       Along with plotting the activation across the surface with time, a
%       second plot is generated showing the "pseudo-electrogram". This reproduces one
%       of the plots from the original paper. (See paper cited above for details.)
%
%       To see some interesting activation patterns, try setting initial state of random 
%       number generator (4th line of code below) to:
%       84 (colliding rotors then random)
%       97 (moving focus, terminates)
%       119 (moving focus/rotor)
%       136 (moving focus)
%       137(multiple wavelets)
%       142(large rotor) 
setting = 84;

R=31; C=32;					    	    % Array dimensions (units).
kv=sqrt(10:20);					        % Range of constants (steps^0.5).
rand('seed',setting);
ik=ceil(size(kv,2)*rand(R,C));		    % Matrix of uniform random numbers, 1:11.
k=kv(ik);							    % Matrix of random constants (steps^0.5).

dt=5;								    % Time step (msec).
arp=-6*ones(size(k));           	    % Initial refractory state throughout (excitable).
nhood=[-1 0;-1 1;0 -1;0 1;1 -1;1 0];	% Neighborhood of adjacent units.
s0=zeros(R,C);				    	    % Array with ones if in state 0.
atimes=cell(R*C,1);			    	    % Cell array of beat times (in steps).

br=[1:31 1:31 zeros(1,34) 32*ones(1,34)];   % Row coordinates of border.
bc=[zeros(1,31) 33*ones(1,31) 0:33 0:33];   % Column coordinates of border.
px=0.5*(br-1)+bc; py=1+(br-1)*sqrt(3)/2;    % Convert coords to hexagonal for plotting.
fh=figure('position',[160 325 350 175],'color',[1 1 1]); % [160 325 630 330]
plot(px,py,'w.','markersize',16); hold on   % Plot the border.
axis image
axh=gca;
set(axh,'xcolor',[1 1 1],'ycolor',[1 1 1])
th=text(15,-2,'');

sr=[12 12 13 13]'; sc=[8 9 8 9]';       % Coordinates of units to stimulate.
si=sub2ind(size(s0),sr,sc);             % Indices of units in cluster to stimulate.
sched=[si ones(size(si))];				% Schedule the units to activate at step 1.
ph1=plot(-2,0,'b.','markersize',16);  % Create handles for active points.
ph2=plot(-2,0,'c.','markersize',16);  % Create handles for refractory points.
for step=1:600
   ia=sched(sched(:,2) == step,1);				% Indices into s0 to activate.
   ia2=sched(sched(:,2) > step,1);              % Additional indices into s0 to plot.
   [sy,sx]=ind2sub(size(s0),[ia;ia2]);          % Coordinates of activated units.
   px=0.5*(sy-1)+sx; py=1+(sy-1)*sqrt(3)/2;     % Convert coords to hexagonal for plotting.
   set(ph1,'xdata',px,'ydata',py);              % Update plot of active units.
   if step > 1
       [ry,rx]=ind2sub(size(s0),inz);               % Coordinates of refractory units.
       px=0.5*(ry-1)+rx; py=1+(ry-1)*sqrt(3)/2;     % Convert coords to hexagonal for plotting.
       set(ph2,'xdata',px,'ydata',py);              % Update plot of refractory units.
       drawnow
       pause(0.025)
       egram(step)=size(inz,1);                     % Build approx egram based on # of active units.
   end
   set(th,'string',num2str(step))                   % Update timer.
   
   s0=zeros(R,C); s0(ia)=1;			        % Update matrix of ones/zeros with activated units.
   sched(sched(:,2) == step,:)=[];          % Remove activated units from table.
   for j=1:size(ia,1)                       % Loop thru units acitvated at this step.
       atimes{ia(j)}=[atimes{ia(j)}; step]; % Append row of atimes with current step.
       if size(atimes{ia(j)},1) > 1         % Element has been activated at least twice.
           lastcycle=diff(atimes{ia(j)}(end-1:end)); %*dt;    % Previous interval (msec).
       else
           lastcycle=40;                    % Use default previous interval of 40 steps.
       end
       arp(ia(j))=round(k(ia(j))*sqrt(lastcycle));   % Set ref pd for units just active.
   end

   % Determine units to schedule for future activation.
   ins0=[];                                     % Indices of neighbors of active units.
   for jj=1:size(nhood,1)	                    % Loop through nhood building column of neighbors.
      ins0=[ins0; find(matrixShift(s0,nhood(jj,1),nhood(jj,2)))];
   end
   if step < 80
       srp=arp(si);
       ins0=[ins0; si(srp<1)];
   end
   ins0=unique(ins0);	        	            % Exclude redundancies.
   ins0=setdiff(ins0,ia);                       % Exclude currently active elements.
   ins0=setdiff(ins0,sched(:,1));               % Exclude units already scheduled for activation.
   inz=find(arp > 0); ins0=setdiff(ins0,inz);   % Exclude units in state 1 (absolute refractory).
   ia2=intersect(ins0,find((arp == 0) | (arp == -1)));	% Units in state 2.
   sched=[sched;[ia2 step+4*ones(size(ia2))]];          % Schedule to activate in 4 steps.
   ia3=intersect(ins0,find((arp == -2) | (arp == -3)));	% Units in state 3.
   sched=[sched;[ia3 step+3*ones(size(ia3))]];          % Schedule to acivate in 3 steps.
   ia4=intersect(ins0,find((arp == -4) | (arp == -5)));	% Units in state 4.
   sched=[sched;[ia4 step+2*ones(size(ia4))]];          % Schedule to acivate in 2 steps.
   ia5=intersect(ins0,find(arp == -6));		            % Units in state 5 (completely recovered).
   sched=[sched;[ia5 step+ones(size(ia5))]];            % Schedule to acivate at next step.
   arp=max(-6,arp-1);                                   % Update refractory period.
end

figure; plot(egram);
title('Pseudo-Electrogram (see paper by Moe et al)')
xlabel('Time Steps')
ylabel('Number of Units Refractory')

% =================== Sub functions ===================
function out=matrixShift(m,r,c)

% MATRIXSHIFT Shift all elements of matrix m up by r rows and right
%   by c columns. Rows and/or columns of zeros are used to fill voids.

for k=1:abs(r)
   if r > 0, m=[m(2:end,:); zeros(1,size(m,2))];
   else m=[zeros(1,size(m,2)); m(1:end-1,:)];
   end
end

for k=1:abs(c)
   if c > 0, m=[zeros(size(m,1),1) m(:,1:end-1)];
   else m=[m(:,2:end) zeros(size(m,1),1)];
   end
end
out=m;