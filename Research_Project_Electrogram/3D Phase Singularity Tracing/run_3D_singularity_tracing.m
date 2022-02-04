%% load data
load p2.mat;
load Virt.dat;
load t.mat;

%% project 3D geometry onto 2D surface
% define discretization size
% going lower than 1 takes way too long, and the 2D projection is not accurate
dx = 5;
dy = 5;

% find out range of x/y/z coords
p = p2;

range_x = range(p(:,1)); % x
range_y = range(p(:,2)); % y
range_z = range(p(:,3)); % z

% make p coords positive
min_x = min(p(:,1)); max_x = max(p(:,1));
min_y = min(p(:,2)); max_y = max(p(:,2));

% convert coordinates to indices based on dx/dy
p(:,1) = round((p(:,1) - min_x)./dx) + 1;
p(:,2) = round((p(:,2) - min_y)./dy) + 1;

% find mid point of geometry in z direction
mid_z = mean(p(:,3));

% convert p to integer
p = int8(p);

% initialize matrix for 2D projection of signal (one for z >= mid, one for z < mid)
Virt2D_1 = nan(ceil(range_x/dx)+1,ceil(range_y/dy),size(Virt,1));
Virt2D_2 = nan(ceil(range_x/dx)+1,ceil(range_y/dy)+1,size(Virt,1));

% initialize matrix to store the original x/y/z coords
loc2D_1 = nan(ceil(range_x/dx)+1,ceil(range_y/dy),3);
loc2D_2 = nan(ceil(range_x/dx)+1,ceil(range_y/dy)+1,3);

% assign values from 3D matrix to 2D, split atria into 2 halves at the middle
for i = 1:size(p,1)
    if p(i,3) >= mid_z
        Virt2D_1(p(i,1),p(i,2),:) = Virt(:,i);  % assign all time steps at once
		loc2D_1(p(i,1),p(i,2),:) = p2(i,:);     % assign x/y/z coords to loc matrix
    else
        Virt2D_2(p(i,1),p(i,2),:) = Virt(:,i);
		loc2D_2(p(i,1),p(i,2),:) = p2(i,:);     % assign x/y/z coords to loc matrix
    end
end

% join 2 halves together
% the left is the bottom half of the atria (from midpoint in z axis)
% the right is the top half
Virt2D = [flip(Virt2D_1,2), Virt2D_2];
loc2D = [flip(loc2D_1,2), loc2D_2];
clear Virt2D_1 Virt2D_2 loc2D_1 loc2D_2

% view the projected geometry at some time step
temp1 = Virt2D(:,:,floor(size(Virt2D,3)/2))';

% view x/y/z coordinates in 2D projection
if 0
   subplot(1,3,1)
   imagesc(loc2D(:,:,1)');colormap('jet');colorbar;title('X position')
   subplot(1,3,2)
   imagesc(loc2D(:,:,2)');colormap('jet');colorbar;title('Y position')
   subplot(1,3,3)
   imagesc(loc2D(:,:,3)');colormap('jet');colorbar;title('Z position')
end

%% smooth 2D projection (interpolate in the center)
% remove middle missing rows
missing_rows = find(sum(1-isnan(Virt2D(:,:,floor(size(Virt2D,3)/2))),1)==0);
Virt2D(:,missing_rows,:) = [];
loc2D(:,missing_rows,:) = [];

% find inner area as binary image
inner = Virt2D(:,:,floor(size(Virt2D,3)/2));
inner = 1-isnan(inner);
k = floor(10/dx);
inner = imclose(inner,strel('disk',k));
inner = logical(inner);

% find min value so we can replace it later
mean_Virt = mean(mean(mean(Virt2D)));

% fill in missing gaps in the middle with interpolation
for i = 1:size(Virt2D,3)
    temp = Virt2D(:,:,i);           % extract 2D slice
    temp(~inner) = mean_Virt;       % locate inner region
    temp = inpaint_nans(temp);      % only fill inner region
    temp(~inner) = nan;             % fill outter border with nans again
    Virt2D(:,:,i) = temp;
end

% repeat for location matrix
for i = 1:3
   temp = loc2D(:,:,i);
   temp(~inner) = 0;
   temp = inpaint_nans(temp);
   temp(~inner) = nan;
   loc2D(:,:,i) = temp;
end

% visualize the 2D projection from 3D after smoothing
if 0
    % before and after smoothing
    figure(1)
    subplot(1,2,1);
    imagesc(temp1);
    colormap('Jet');colorbar;
    title('3D projection before smoothing');
	
    subplot(1,2,2);
    imagesc(Virt2D(:,:,floor(size(Virt2D,3)/2))');
    colormap('Jet');colorbar;
    title('3D projection after smoothing');

    % movie
    figure(2)
    for i = 1:size(Virt2D,3)
        imagesc(Virt2D(:,:,i)');
        pause(0.01);fprintf('%d\n',i);
    end
end

%% phase mapping
% generate phase map
[Phase_Map] = ExtraCellularUnipolarPotential2PhaseMap(Virt2D,1000);

% view movie
if 0
    for i = 1:size(Phase_Map,3)
        imagesc(Phase_Map(:,:,i)');
        pause(0.01);fprintf('%d\n',i);
    end
end

%% phase singularity tracing
% locate potential singularities
[PS_trajectory] = LocalizePhaseMapSingularity(Phase_Map);

% find most likely occuring PS, since each timestep contains many potential PS
% we take the current PS, and the next PS is one that is likely closest to
% the last one
PS_2D = zeros(length(PS_trajectory),2);
PS_2D(1,:) = mean(PS_trajectory{1},1);  % first PS is just the mean position

for i = 2:length(PS_trajectory)
    
    % get all calculated PS for this time step
    PS_temp = PS_trajectory{i};
    
    % find which co-ord in the current time step is closest to the last
    % also track which point is closest to all previous PS's to make sure
    % that it doesnt sway too far from the current pattern
    dist1 = sqrt((PS_2D(i-1,1) - PS_temp(:,1)).^2 + (PS_2D(i-1,2) - PS_temp(:,2)).^2);
    dist2 = sqrt((nanmean(PS_2D(1:(i-1),1)) - PS_temp(:,1)).^2 + (nanmean(PS_2D(1:(i-1),2)) - PS_temp(:,2)).^2);
    [~, min_ind] = min(dist1 + dist2);
    
    % assign value to the PS array
    PS_2D(i,:) = PS_temp(min_ind,:);
    
end

% view trajectory over time (progression)
if 0
    delay = 150;
    for i = (delay+500):size(Phase_Map,3)
        
        subplot(1,2,1);
        imagesc(Phase_Map(:,:,i)');colormap('Jet');
        colorbar;
        hold on;
        scatter(PS_2D(i,2),PS_2D(i,1),200,'MarkerEdgeColor','w','MarkerFaceColor','k','linewidth',2.5);
        
        subplot(1,2,2);
        plot(PS_2D((i-delay):i,2),PS_2D((i-delay):i,1),'b-o');
        axis([0 size(Phase_Map,1) 0 size(Phase_Map,2)]);
        set(gca, 'YDir','reverse');
        
        pause(0.01);fprintf('%d\n',i);
    end
end

% view final trajectory over all time steps
if 0
    background = Phase_Map(:,:,i)';
    background(~isnan(background)) = 0;
    imagesc(background);
    title('Trajectory over all time steps');
    hold on;
    plot(PS_2D(:,2),PS_2D(:,1),'-k');
    axis([0 size(Phase_Map,1) 0 size(Phase_Map,2)]);
    set(gca, 'YDir','reverse');
end

%% re-trace coordinatees back to 3D
% initialize coords for 3D
PS_2D = ceil(PS_2D);
PS_3D = nan(size(PS_2D,1),3);

% convert the 2D PS locations back to 3D
for i = 1:size(PS_3D,1)
    if sum(isnan(PS_2D(i,:))) == 0
        PS_3D(i,:) = loc2D(PS_2D(i,2),PS_2D(i,1),:);
    end
end

%% compute singularity heat map in 3D (frequency of occurence)
heat_map = zeros(size(Virt,2),1);

for i = 1:size(PS_3D,1)
    if sum(isnan(PS_3D(i,:))) == 0
        % find ind of position vector containing the PS
        [~,ind] = min(sqrt(sum(bsxfun(@minus,p2,PS_3D(i,:)).^2,2)));

        % increment the index of the heat map vector
        heat_map(ind,1) = heat_map(ind,1) + 1;
    end
end

%% visualize in 3D
close all;

% define coordinates of certain features
a1 = -99.6;     a2 = -16;
LSPVx = -2.51;  LSPVy = -36.97; LSPVz = 4.21;
LIPVx = -0.53;  LIPVy = -28.86; LIPVz = 39.39;
RSPVx = -23.03; RSPVy = 14.65;  RSPVz = -28.65; 
RIPVx = -21.90; RIPVy = 30.13;  RIPVz = -5.53;
LAAx  = 38.25;  LAAy  = -17.72; LAAz  = 5.49;

% plot 3D surface
%trisurf(t,p2(:,1),p2(:,2),p2(:,3),zeros(1,size(Virt,2)),'FaceColor','interp');
trisurf(t,p2(:,1),p2(:,2),p2(:,3),heat_map,'FaceColor','interp','edgecolor','none');
colormap('jet'); colorbar;

% add text
axis equal;
daspect([1 1 1])
text(LSPVy,LSPVx,-LSPVz,' \leftarrow LSPV','FontSize',18)
text(LIPVy,LIPVx,-LIPVz,' \leftarrow LIPV','FontSize',18)
text(RSPVy,RSPVx,-RSPVz,' \leftarrow RSPV','FontSize',18)
text(RIPVy,RIPVx,-RIPVz,' \leftarrow RIPV','FontSize',18)
text(LAAy,LAAx,-LAAz,' \leftarrow LAA','FontSize',18)
view([a1 a2])
rotate3d;

% add trajectory lines
%hold on;
%plot3(PS_3D(:,1),PS_3D(:,2),PS_3D(:,3),'-k');
