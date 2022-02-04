% 12- PDE approach
%Author Aaqel Nalar (Feb 2019) - Adapted from Marko Ruslim

% Skips steps 12b and outputs thickness directly in terms of
% number of pixels

% Run this script 3 times, Once for L0, Once for L1 and Once when both have
% been found

% Note that the PDE convergence solution should take <150 iterations per solution,
% anything significantly over (e.g. 300) indicates anomolous points due to structure
% or holes in tissue

% (O)N^3 function. -> larger data will cause N^3 increase in duration per iteration. Number of iterations depends on nature of data and anomalies.
%% Load data
parent = pwd;
cd(parent)
cd([Heart, '/']);

% section = input('Enter 1 to solve for L1, 0 to solve for L0, -1 if both have been found: ');

load parameters.mat %From Noise Removal (1)
swap = Nx;
Nx = Ny;
Ny = swap;
clear swap

if section ~= -1
    disp('Reading data');
    
    load 3DCavityLAendo.mat %From regionGrowing Processing (5)
    load 3DCavityRAendo.mat %From regionGrowing Processing (5)
    load endoRA_surface_masked.mat %From createSurfaceMasks_new (6b)
    load endoLA_surface_masked.mat %From createSurfaceMasks_new (6b)
    load epi_surface_masked.mat %From createSurfaceMasks_new (6b)
    load tissueInside.mat %From createLayeredTissue (7)
    load outside.mat %From createLayeredTissue (7)
    load TransmuralFieldLines.mat %From LaplaceSolverWithMiddleLine (10)
    %load atriaClosingMask.mat
    load newMiddleLine.mat %From MiddleLine (9)
    Middle = uint8(Middle);
    epi_surface(Middle>0)=1;
    
    % DONT USE MIDDLE REGION MASKS
    %query = input('Is there a middle region mask? 1 for yes, 0 for no: ');
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
    
    
    
    disp('Recreating gradient field');
    load gradientField.mat; % From createGradientField (11)
    Ntissue(:,:,:,1) = gradientFieldX;
    Ntissue(:,:,:,2) = gradientFieldY;
    Ntissue(:,:,:,3) = gradientFieldZ;
    
    clear gradientFieldX gradientFieldY gradientFieldZ
    ZeroMap=uint8(zeros(Ny,Nx,Nz));
    for ii=1:1:Ny
        for jj=1:1:Nx
            for kk=1:1:Nz
                if (Ntissue(ii,jj,kk,1)==0)&&(Ntissue(ii,jj,kk,2)==0)&&(Ntissue(ii,jj,kk,3)==0)
                    ZeroMap(ii,jj,kk)=1;
                end
            end
        end
    end
end
%% Define Lo and L1
% Note : 20/07/16 - Solve Lo and L1 separately
%% Lo Laplace Solver and Coupled PDES

if section == 0
    
    Lo=zeros(Ny,Nx,Nz);
    % Define Solving Region (SR) Lo
    SR = logical(zeros(Ny, Nx, Nz));                  % Solve Region
    SR(tissueInside > 0) = 1;
    %SR(epi > 0) = 1;
    SR(endoLA_surface > 0) = 1;
    SR(endoRA_surface > 0) = 1;
    SR(ZeroMap==1)=0;
    
    if query == 1
        SR(Middle>0 & MiddleRegion>0)=0;
    else
        SR(Middle>0)=0;
    end
    
    
    %     %disp('Manual Adjustment for H158 in Lo');
    %     SR(144,320,157) = 0; % H158 manual adjustment (removes a single spec)
    %     SR(144,319,157) = 0; % H158 manual adjustment (removes a single spec)
    %     SR(145,319,157) = 0; % H158 manual adjustment (removes a single spec)
    %
    
    %% Remove detached regions
    %     dataCleaned = SR;
    %
    %
    %     disp('Removing unconnected regions');
    %     Labels = bwlabeln(dataCleaned,6);
    %     stats = regionprops(Labels,'Area');
    %     CRegions = [stats.Area];
    %     [~,biggest] = max(CRegions);
    %     dataTemp = dataCleaned;
    %     dataTemp(Labels~=biggest) = 0;
    %
    %     dataFirst = dataTemp;
    %
    %     dataCleaned = SR;
    %     dataCleaned(dataFirst>0) = 0;
    %
    %     Labels = bwlabeln(dataCleaned,6);
    %     stats = regionprops(Labels,'Area');
    %     CRegions = [stats.Area];
    %     [~,biggest] = max(CRegions);
    %     dataTemp = dataCleaned;
    %     dataTemp(Labels~=biggest) = 0;
    %
    %     dataSecond = dataTemp;
    %
    %     clear dataTemp dataCleaned
    %
    %     SR = dataFirst + dataSecond;
    
    %%
    tissue=Lo;
    
    clear endoLA_surface endoLACavity endoRA_surface endoRACavity epi_surface Middle tissueInside
    
    %     w = 1.4;
    %     disp(['using relaxation factor: ', num2str(w)]);
    
    h = 1;
    iter_max = 200;
%     input('Enter the maximum number of iterations (200 recommended): ');        % maximum iterations
    accuracy = 0.1; %1e-5;          % desired relative error (%)
    disp(['Performing Laplace Solution for Lo with an accuracy of ', num2str(accuracy), '%']);
    pause(2);
    iter = 0;
    max_rel_err = inf;     % placeholder value
    % T = Ntissue;
    intermediate=[];
    [indY, indX, indZ] = ind2sub(size(SR), find(SR > 0) );
    while (max_rel_err>=accuracy) && (iter<iter_max)
        tic
        %state=0
        %while (state==0),
        iter = iter + 1;
        max_rel_err = 0;
        lastPercentage = 0;
        for i = 1 : length(indY)
            iy = indY(i);
            ix = indX(i);
            iz = indZ(i);
            
            percentage = round(100*i/length(indY));
            if percentage ~= lastPercentage
                lastPercentage = percentage;
                if percentage >35
                    disp(['percentage of iteration ', num2str(iter), ' complete: ',num2str(percentage),'%']);
                end
            end
            if (SR(iy,ix,iz) > 0)
                gradY=Ntissue(iy,ix,iz,1);
                gradX=Ntissue(iy,ix,iz,2);
                gradZ=Ntissue(iy,ix,iz,3);
                
                if gradY>0
                    DyLo=tissue(iy-1,ix,iz);
                elseif gradY<=0
                    DyLo=tissue(iy+1,ix,iz);
                end
                
                if gradX>0
                    DxLo=tissue(iy,ix-1,iz);
                elseif gradX<=0
                    DxLo=tissue(iy,ix+1,iz);
                end
                
                if gradZ>0
                    DzLo=tissue(iy,ix,iz-1);
                elseif gradZ<=0
                    DzLo=tissue(iy,ix,iz+1);
                end
                
                gradY=abs(gradY);
                gradX=abs(gradX);
                gradZ=abs(gradZ);
                
                
                temp_old = tissue(iy,ix,iz);
                
                denom=( gradX + gradY + gradZ );
                
                temp_int = ( 1 + gradX*DxLo + gradY*DyLo + gradZ*DzLo) / denom;
                
                %temp_new = (1-w)*temp_old + w*temp_int;
                temp_new = temp_int;
                
                rel_err = abs((temp_new - temp_old)/temp_new)*100;
                
                tissue(iy,ix,iz) = temp_new;
                
                if (rel_err>max_rel_err)
                    max_rel_err = rel_err;
                end
                
                clear DxLo;clear DyLo; clear DzLo;
                
            end
        end
        
        % display progress
        disp(['i:',num2str(iter), '/', num2str(iter_max),' R.E.:',num2str(max_rel_err),'%.  Target -> ', num2str(accuracy), '%']);
        if (mod(iter,10)==0)
%             figure(1)
%             imagesc(tissue(:,:,round(Nz/2)));
%             colorbar();
            disp(['Saving result at iter: ', num2str(iter)]);
            Lo=tissue;
            save Lo.mat Lo max_rel_err iter
        end
        
        if (max_rel_err<accuracy)
            disp([num2str(100-accuracy), sprintf('%% solved after %0.f iterations',iter)]);
        elseif (iter==iter_max)
            disp([num2str(abs(100-max_rel_err)), sprintf('%% solved after %0.f iterations',iter_max)]);
            
            % Anomaly Finder
            disp('Anomalies present at: ');
            for i = 1:Nz
                if (max(max(Lo(:,:,i)))>150) % 150 used as a threshold saying anything high is an anomaly
                    disp(num2str(i));
                end
            end
            break;
        end
        toc
    end
    
    % figure,
    % plot((1:iter),squeeze(intermediate(138,99,:)))
    %save Intermediate.mat intermediate
    %return
    disp('Lo Solved');
    Lo=tissue;
    clear tissue;
    save -v7.3 Lo.mat Lo iter max_rel_err
    
    
    
elseif section == 1
    
    
    %% L1
    % Define Solving Region (SR)
    L1=zeros(Ny,Nx,Nz);
    SR = uint8(zeros(Ny, Nx, Nz));                  % Solve Region
    SR(tissueInside > 0) = 1;
    SR(epi_surface > 0) = 1;
    %SR(endoLA > 0) = 1;
    %SR(endoRA > 0) = 1;
    SR(ZeroMap==1)=0;
    if query == 1
        SR(Middle>0 & MiddleRegion>0)=0;
    else
        SR(Middle>0)=0;
    end
    clear endoLA_surface endoLACavity endoRA_surface endoRACavity epi_surface Middle tissueInside
    %SR(VFtissue==110)=0;
    
    %% Remove detached regions
    %     dataCleaned = SR;
    %
    %
    %     disp('Removing unconnected regions');
    %     Labels = bwlabeln(dataCleaned,6);
    %     stats = regionprops(Labels,'Area');
    %     CRegions = [stats.Area];
    %     [~,biggest] = max(CRegions);
    %     dataTemp = dataCleaned;
    %     dataTemp(Labels~=biggest) = 0;
    %
    %     dataFirst = dataTemp;
    %
    %     dataCleaned = SR;
    %     dataCleaned(dataFirst>0) = 0;
    %
    %     Labels = bwlabeln(dataCleaned,6);
    %     stats = regionprops(Labels,'Area');
    %     CRegions = [stats.Area];
    %     [~,biggest] = max(CRegions);
    %     dataTemp = dataCleaned;
    %     dataTemp(Labels~=biggest) = 0;
    %
    %     dataSecond = dataTemp;
    %
    %     clear dataTemp dataCleaned
    %
    %     SR = dataFirst + dataSecond;
    
    %%
    
    tissue=L1;
    
    %w = 1.4;
    %disp(['using relaxation factor: ', num2str(w)]);
    
    h = 1;
    iter_max = 200;     % maximum iterations
    accuracy = 0.1; %1e-5;          % desired relative error (%)
    disp(['Performing Laplace Solution for L1 with an accuracy of ', num2str(accuracy), '%']);
    iter = 0;
    pause(2);
    
    max_rel_err = inf;     % placeholder value
    % T = Ntissue;
    [indY, indX, indZ] = ind2sub(size(SR), find(SR > 0) );
    while (max_rel_err>=accuracy) && (iter<iter_max)
        tic
        %state=0
        %while (state==0),
        iter = iter + 1;
        max_rel_err = 0;
        lastPercentage = 0;
        for i = 1 : length(indY)
            iy = indY(i);
            ix = indX(i);
            iz = indZ(i);
            
            percentage = round(100*i/length(indY));
            if percentage ~= lastPercentage
                lastPercentage = percentage;
                if percentage >35
                    disp(['percentage of iteration ', num2str(iter), ' complete: ',num2str(percentage),'%']);
                end
            end
            if (SR(iy,ix,iz) > 0)
                
                gradY=Ntissue(iy,ix,iz,1);
                gradX=Ntissue(iy,ix,iz,2);
                gradZ=Ntissue(iy,ix,iz,3);
                
                if gradY>0
                    DyLo=tissue(iy+1,ix,iz);
                elseif gradY<=0
                    DyLo=tissue(iy-1,ix,iz);
                end
                
                if gradX>0
                    DxLo=tissue(iy,ix+1,iz);
                elseif gradX<=0
                    DxLo=tissue(iy,ix-1,iz);
                end
                
                if gradZ>0
                    DzLo=tissue(iy,ix,iz+1);
                elseif gradZ<=0
                    DzLo=tissue(iy,ix,iz-1);
                end
                
                gradY=abs(gradY);
                gradX=abs(gradX);
                gradZ=abs(gradZ);
                
                temp_old = tissue(iy,ix,iz);
                
                temp_int = ( 1 + gradX*DxLo + gradY*DyLo + gradZ*DzLo) / ( gradX + gradY + gradZ );
                
                %temp_new = (1-w)*temp_old + w*temp_int;
                temp_new = temp_int;
                
                rel_err = abs((temp_new - temp_old)/temp_new)*100;
                
                tissue(iy,ix,iz) = temp_new;
                
                if (rel_err>max_rel_err)
                    max_rel_err = rel_err;
                end
                clear DxLo;clear DyLo; clear DzLo;
            end
            
        end
        
        % display progress
        disp(['i:',num2str(iter), '/', num2str(iter_max),' R.E.:',num2str(max_rel_err),'%.  Target -> ', num2str(accuracy), '%']);
        if (mod(iter,10)==0)
%             figure(1)
%             imagesc(tissue(:,:,round(Nz/2)));
%             colorbar();
            disp(['Saving result at iter: ', num2str(iter)]);
            L1=tissue;
            save -v7.3 L1.mat L1 max_rel_err iter
        end
        
        if (max_rel_err<accuracy)
            disp([num2str(100-accuracy), sprintf('%% solved after %0.f iterations',iter)]);
        elseif (iter==iter_max)
            disp([num2str(abs(100-max_rel_err)), sprintf('%% solved after %0.f iterations',iter_max)]);
            
            % Anomaly Finder
            disp('Anomalies present at: ');
            for i = 1:Nz
                if (max(max(L1(:,:,i)))>150)
                    disp(num2str(i));
                end
            end
            
            break;
        end
        toc
        %imagesc(tissue(:,:,150))
        %drawnow
    end
    
    L1=tissue;
    clear tissue;
    save -v7.3 L1.mat L1 max_rel_err iter
    disp('L1 Solved');
    
    
elseif section == -1
    %% Solve for thickness
    %date = input('Enter the date as a string including apostrophes (e.g. ''11Feb''): ');
%     load parameters.mat
    load L1.mat %From AWT PDE MarkoVersion (12)
    load Lo.mat %From AWT PDE MarkoVersion (12)
    disp('Calculating Thickness');
    Thickness=L1+Lo;
    
    %% Unmasking
    
    load BinaryAtriaOnly.mat %From Morphological Operations (3)
    se = strel('ball',3,1);
    for i = 1:Nz
        I2 = imdilate(squeeze(Thickness(:,:,i)),se);
    end
    
    disp('Unmasking Surfaces: MAKE SURE CORRECT ATRIA CLOSING MASK IS USED');
    % Read mask data
    mask = zeros(Ny,Nx,Nz, 'uint8');
    maskFiles = dir([maskPath '*.tif']);
    for i=1:Nz
        mask(:,:,i) = imread([maskPath maskFiles(i).name]);
    end
    
    cd(parent);
    
    mask2 = growByOne(mask);
    mask3 = growByOne(mask2);
    
    Lo(mask3 ==2 & tempCleanFilled == 0)=0; % only the artificial main holes, not structural holes
    L1(mask3 ==2 & tempCleanFilled == 0)=0; % only the artificial main holes, not structural holes
    
    L0(mask3 > 0)=0; % only the artificial main holes, not structural holes
    L1(mask3 > 0)=0; % only the artificial main holes, not structural holes
    
%     disp('creating images');
%     figure,imagesc(Lo(:,:,100))
%     pbaspect([Nx Ny 1]);
%     colormap(jet)
%     %caxis([0 25])
%     
%     figure,imagesc(L1(:,:,100))
%     pbaspect([Nx Ny 1]);
%     colormap(jet)
%     %caxis([0 25])
    
    Thickness_unmasked=L1+Lo;
    
%     %% View figures
%     disp('plotting figures');
%     % Plot thickness map in XY plane with grid to show zoomed in view
%     figure,imagesc(Thickness(:,:,100))
%     pbaspect([Nx Ny 1]);
%     colormap(jet)
%     caxis([0 10])
%     set(gca,'xtick', linspace(0.5,Nx+0.5,Nx+1), 'ytick', linspace(0.5,Ny+.5,Ny+1));
%     set(gca,'xgrid', 'on', 'ygrid', 'on', 'gridlinestyle', '-', 'xcolor', 'k', 'ycolor', 'k');
%     
%     % Plot thickness map in XZ plane with grid to show zoomed in view
%     figure,imagesc(squeeze(Thickness(281,:,:)))
%     pbaspect([Nz Nx 1]);
%     colormap(jet)
%     caxis([0 10])
%     set(gca,'xtick', linspace(0.5,Nz+0.5,Nz+1), 'ytick', linspace(0.5,Nx+.5,Nx+1));
%     set(gca,'xgrid', 'on', 'ygrid', 'on', 'gridlinestyle', '-', 'xcolor', 'k', 'ycolor', 'k');
%     
%     % Plot thickness map in YZ plane with grid to show zoomed in view
%     figure,imagesc(squeeze(Thickness(:,310,:)))
%     pbaspect([Nz Ny 1]);
%     colormap(jet)
%     caxis([0 10])
%     set(gca,'xtick', linspace(0.5,Nz+0.5,Nz+1), 'ytick', linspace(0.5,Ny+.5,Ny+1));
%     set(gca,'xgrid', 'on', 'ygrid', 'on', 'gridlinestyle', '-', 'xcolor', 'k', 'ycolor', 'k');
%     
    
    %% Save data
    
    cd([Heart, '/']);
    disp('saving data');
    save(['AWT_', Heart, '_', '_PDE.mat'] ,'Thickness_unmasked', 'L1', 'Lo', 'Thickness','-v7.3');
    save(['Thickness.mat'], 'Thickness', 'Thickness_unmasked','-v7.3');
    disp('Complete');
    disp('Create an apppendage mask in amira and then continue to WallThicknessResults (13)');
    
else
    disp('invalid number');
    return
end