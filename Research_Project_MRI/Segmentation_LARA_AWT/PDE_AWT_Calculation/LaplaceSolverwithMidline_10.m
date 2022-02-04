%8 and %10
% Author: Aaqel Nalar (Jan 2019)
%% Laplace solver

parent = pwd;
cd(parent)
cd([Heart, '/']);

%% Load data information

disp('reading data');
load parameters.mat; %From Noise Removal (1)
%% Do Not use middle region masks -> query permanently set to 0
query = 0;
if query == 1
    path = './MiddleRegion/';
    files = dir([path '*.tif']);
    
    %Get the dimensions needed for the loops by checking the first file
    Nz = length(files);
    MiddleRegion(:,:,1) = imread([path files(1).name]);
    [Nx,Ny] = size(MiddleRegion(:,:,1));
    %Read the rest of the images
    for i=2:Nz
        MiddleRegion(:,:,i) = imread([path files(i).name]);
    end
end

swap = Nx;
Nx = Ny;
Ny = swap;
clear swap

load tissueInside.mat; % From createLayeredTissue (7)
load epi_surface_masked.mat; % From createSurfaceMasks_New (6)
load endoRA_surface_masked.mat; % From createSurfaceMasks_New (6b)
load endoLA_surface_masked.mat; % From createSurfaceMasks_New (6b)
load 3DCavityLAendo.mat; % From regionGrowingProcessing (5)
load 3DCavityRAendo.mat; % From regionGrowingProcessing (5)

endoRACavity = uint8(endoRACavity);
endoLACavity = uint8(endoLACavity);

% ensure no overlap
tissueInside(endoRA_surface > 0)   = 0;
tissueInside(endoLA_surface > 0)   = 0; 
tissueInside(epi_surface > 0) = 0;
tissueInside(endoRACavity>0)=0;
tissueInside(endoLACavity>0)=0;



%% FIRST RUN TO FIND MID LINE
% tissue = ones(Ny, Nx, Nz)*200; % everywhere else is zeros %% might have to change this because hard to measure the relative error
% tissue(HollowEndoRA > 0)   = 300; %endo of LA = 100
% tissue(HollowEndoLA > 0)   = 100; %endo of RA = -100
% tissue(Hollow_epi > 0) = 200;
% tissue(endoRACavity>0)=300;
% tissue(endoLACavity>0)=100;


%% SECOND RUN AFTER GETTING MIDDLE LINE

load MiddleLine.mat % From MiddleLine (9)

tissue = ones(Ny, Nx, Nz)*100;
tissue(endoRACavity>0)= 300;
tissue(endoLACavity>0)= 300;
tissue(endoRA_surface > 0) = 300; %endo of LA = 100
tissue(endoLA_surface > 0) = 300;

LAPSR = zeros(Ny, Nx, Nz);                  % LAPlace Solve Region (LAPSR)
LAPSR(tissueInside > 0) = 1;
if query == 1
    LAPSR(Middle>0 & MiddleRegion > 0)=0;
else
    LAPSR(Middle>0)=0;
end

% To remove regions which are cut and seperated by the mid line

% Original = LAPSR;
% dataCleaned = LAPSR;
% 
% disp('Removing unconnected regions');
% Labels = bwlabeln(dataCleaned,6);
% stats = regionprops(Labels,'Area');
% CRegions = [stats.Area];
% [~,biggest] = max(CRegions);
% dataTemp = dataCleaned;
% dataTemp(Labels~=biggest) = 0;
% 
% dataFirst = dataTemp;
% 
% dataCleaned = LAPSR;
% dataCleaned(dataFirst>0) = 0;
% 
% Labels = bwlabeln(dataCleaned,6);
% stats = regionprops(Labels,'Area');
% CRegions = [stats.Area];
% [~,biggest] = max(CRegions);
% dataTemp = dataCleaned;
% dataTemp(Labels~=biggest) = 0;
% 
% dataSecond = dataTemp;
% 
% clear dataTemp dataCleaned
% 
% LAPSR = dataFirst + dataSecond;
% New = LAPSR;

tissue(LAPSR > 0) = 0;
%%
% 
% % used to be different -> that is saved as outsideWithMiddleLine.mat
% outside = uint8(ones(Ny,Nx,Nz));
% outside(endoLACavity>0) = 0;
% outside(endoRACavity>0) = 0;
% outside(Hollow_epi>0) = 0;
% outside(HollowEndoLA>0) = 0;
% outside(HollowEndoRA>0) = 0;
% outside(tissueInside>0) = 0;

%% Laplace Solver
% relaxation factor
tic
w = 1.4;
disp(['using relaxation factor of : ', num2str(w)]);
cd % print working directory
h = 1;
  
iter_max = 1500;
% input('Enter the maximum number of iterations (1500 recommended): ');        % maximum iterations
accuracy = 0.1;          % desired relative error (%)
disp(['Performing Laplace Solution with an accuracy of ', num2str(accuracy), '%']);
iter = 0;  
max_rel_err = inf;     % placeholder value


while (max_rel_err>=accuracy) && (iter<iter_max)
    iter = iter + 1;
    max_rel_err = 0;
    for iz = 2:1:(Nz-1)
        for iy = 2:1:(Ny-1)
            for ix = 2:1:(Nx-1)
                if (tissueInside(iy,ix,iz) > 0)
       
                    temp_old = tissue(iy,ix,iz);
                    
                    temp_int = (tissue(iy+h,ix,iz) + tissue(iy-h,ix,iz) ... 
                              + tissue(iy,ix+h,iz) + tissue(iy,ix-h,iz) ...
                              + tissue(iy,ix,iz+h) + tissue(iy,ix,iz-h))/6.0;
                    
                    temp_new = (1-w)*temp_old + w*temp_int;
                    
                    rel_err = abs((temp_new - temp_old)/temp_new)*100;
                    
                    tissue(iy,ix,iz) = temp_new;
                     
                    
                    if (rel_err>max_rel_err)
                 
                        max_rel_err = rel_err;
                       
                    end
                end
            end
        end
    end
    % display progress
        disp(['i:',num2str(iter), '/', num2str(iter_max),' R.E.:',num2str(max_rel_err),'%.  Target -> ', num2str(accuracy), '%']);
    
    if (max_rel_err<accuracy)
        disp([num2str(100-accuracy), sprintf('%% solved after %0.f iterations',iter)]);
    elseif (iter==iter_max)
        disp([num2str(abs(100-max_rel_err)), sprintf('%% solved after %0.f iterations',iter_max)]);
        break;
    end

end
toc
 %% Save initial potential field for finding midline
% 
% save EndoToEndoLaplaceFieldLines.mat tissue

%% Second iteration field lines for transmurality
disp('saving transmural laplace solution');
% save outside.mat outside

% disp('Manual Adjustment for H158');
% tissue(:,:,95) = zeros(Ny,Nx);
save -v7.3 TransmuralFieldLines.mat tissue
disp('Complete');
disp('Run Gradient Field Script');