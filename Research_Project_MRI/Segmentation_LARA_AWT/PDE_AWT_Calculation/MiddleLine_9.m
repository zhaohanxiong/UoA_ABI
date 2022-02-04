% 9
%% MiddleLine
% Author: Aaqel Nalar (Jan 2019)
parent = pwd;
cd(parent)
cd([Heart, '/']);

disp('Loading data');
load EndoToEndoLaplaceFieldLines.mat; %From Laplace Solver (8)
load parameters.mat; % From Noise Removal (1)

% swap x and y to comply with following code
swap = Nx;
Nx = Ny;
Ny = swap;
clear swap

load tissueInside.mat; % From labelled data (7)
load epi_surface_masked.mat; % From createSurfaceMasks_New (6b)
load endoRA_surface_masked.mat; % From createSurfaceMasks_New (6b)
load endoLA_surface_masked.mat; % From createSurfaceMasks_New (6b)
load 3DCavityLAendo.mat; % From regionGrowingProcessing (5)
load 3DCavityRAendo.mat; % From regionGrowingProcessing (5)

% endoRACavity = uint8(endoRACavity);
% endoLACavity = uint8(endoLACavity);

% ensure no overlap
tissueInside(endoRA_surface > 0)   = 0;
tissueInside(endoLA_surface > 0)   = 0; 
tissueInside(epi_surface > 0) = 0;
tissueInside(endoRACavity>0)=0;
tissueInside(endoLACavity>0)=0;

%% 200 is the middle line 
disp('Computing Middle Line');
tissueMask=tissueInside;
tissue(tissue==200)=0;

%% Grow RA
cd(parent)
tic
disp('Growing Right Atria');
RA = tissue;
RA (tissue>=200)=0;
seed = [1 1 1; Ny 1 1; 1 Nx 1; 1 1 Nz; Ny Nx Nz; Ny Nx 1; 1 Nx Nz; 1 1 Nz];
tissue0=RA;



[region, surface] = MyRegionGrowFunc(tissue0, seed, 0, -1);
epi_surface = surface;      % epicardium
outside = region;   % outside space
RA=epi_surface-outside;
toc
%% Grow LA
tic
disp('Growing Left Atria');
LA = tissue;
LA (tissue<=200)=0;
seed = [1 1 1; Ny 1 1; 1 Nx 1; 1 1 Nz; Ny Nx Nz; Ny Nx 1; 1 Nx Nz; 1 1 Nz];
tissue0=LA;
[region, surface] = MyRegionGrowFunc(tissue0, seed, 0, -1);
epi = surface;      % epicardium
outside = region;   % outside space
LA=epi-outside;
toc
%% Calculate total episurface


disp('showing epicardial');

epi=100*RA+200*LA;
% figure,imagesc(epi(:,:,150))

%% Compute Middle Line and check

disp('showing Midline');

Middle=epi;
Middle(tissueMask==0)=0;
% figure,imagesc(Middle(:,:,150))


%% Save data
[Nx,Ny,Nz] = size(Middle);
newmid = zeros(Nx,Ny,Nz);
for i = 1:Nx-1
    for j = 1:Ny-1
        for k = 1:Nz
            if Middle(i,j,k) == 200
                if Middle(i+1,j,k) == 100
                    newmid(i,j,k) = Middle(i,j,k);
                    newmid(i+1,j,k) = Middle(i+1,j,k);
                elseif Middle(i,j+1,k) == 100
                    newmid(i,j,k) = Middle(i,j,k);
                    newmid(i,j+1,k) = Middle(i,j+1,k);
                    
                end
            end
        end
    end
end
cd([Heart, '/']);
disp('Saving Midline');

save -v7.3 MiddleLine.mat Middle

Middle = newmid;               

save -v7.3 newMiddleLine.mat Middle
disp('Complete');
disp('Run Laplace Solver (10) Second Run');
