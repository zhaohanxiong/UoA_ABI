%14
%% get AWT colormap
clear


parent = pwd;
Heart = input('Enter Heart Number/ Folder name as string including apostrophes (e.g. ''H109''): ');
date = input('Enter the date as a string including apostrophes (e.g. ''11Feb''): ');
cd(parent)
cd([Heart, '/']);

disp('Make sure data for AWTmap is correct in code: ');
AWTmap = 'Unrounded_AWT_CARMA1421_pre_results_test_PDE.mat' % change accordingly
load(AWTmap);
load parameters.mat
%% Get into grayscale uint8 form
disp('converting to grayscale');
dataToConvert = AWTfinal;

AWTcolormap = (dataToConvert/max(max(max(AWTfinal))))*255; %% This is an absolute max value so colors can be compared
AWTcolormap = uint8(AWTcolormap);
clear AWTfinal;
%% GrowSurfaces

disp('Loading surfaces');

load endoLA_surface.mat
load endoRA_surface.mat
load epi_surface.mat % for surface itself
% load appendageMask.mat % for surface colormap
load BinaryAtriaOnly.mat

epiSurface_noAppendage = tempCleanFilled;
% epiSurface_noAppendage(maskAppendage > 0)=0;

for mode = 1:4
    if mode == 1
        disp('Growing tempCleanFilled');
        dataToGrow = tempCleanFilled;
    elseif mode == 2
        disp('Growing endoLA');
        dataToGrow = endoLA;
    elseif mode == 3
        disp('Growing endoRA');
        dataToGrow = endoRA;
    elseif mode == 4
        disp('Growing tempCleanFilled colormap mask')
        dataToGrow = epiSurface_noAppendage;
    end
    
    dataToMask = AWTcolormap;
    
    %% Grow to dataGrown14 for tempCleanFilled, dataGrown10 for endo
    
    %grow to 14 for tempCleanFilled
    if mode == 1 || mode == 4
        iter = 14;
        
        %grow to 10 for endo
    else
        iter = 10;
    end
    
    cd(parent)
    
    disp(['growing to thickness of ', num2str(iter) ' pixels']);
    
    dataGrownOld = dataToGrow;
    clear dataToGrow
    
    % grow regoins
    for i = 2:iter
        

        dataGrownNew = growByOne(dataGrownOld);
        disp([num2str(i), ' out of ' num2str(iter) ' pixels grown']);
        
        if i == 2
            %save first region
            dataGrown2 = dataGrownNew;
        end
        
        if i == 3
            %save first region (for epi)
            dataGrown3 = dataGrownNew;
        end
        
        % start masking off regions which are not present
        if i > 4
            dataGrownNew(dataToMask == 0) = 0;
        end
        
        % save last region
        if i ~= iter
            dataGrownOld = dataGrownNew;
        end
        
    end
    
    
    %     dataGrown2 = growByOne(dataToGrow);
    %     %dataGrown2(dataToMask==0)=0;
    %
    %     dataGrown3 = growByOne(dataGrown2);
    %     %dataGrown3(dataToMask==0)=0;
    %
    %     dataGrown4 = growByOne(dataGrown3);
    %     %dataGrown4(dataToMask==0)=0;
    %
    %     dataGrown5 = growByOne(dataGrown4);
    %     dataGrown5(dataToMask==0)=0;
    %
    %
    %     dataGrown6 = growByOne(dataGrown5);
    %     dataGrown6(dataToMask==0)=0;
    %
    %     dataGrown7 = growByOne(dataGrown6);
    %     dataGrown7(dataToMask==0)=0;
    %
    %     % For tempCleanFilled only
    %     dataGrown8 = growByOne(dataGrown7);
    %     dataGrown8(dataToMask==0)=0;
    %
    %     dataGrown9 = growByOne(dataGrown8);
    %     dataGrown9(dataToMask==0)=0;
    %
    %     dataGrown10 = growByOne(dataGrown9);
    %     dataGrown10(dataToMask==0)=0;
    %
    %     dataGrown11 = growByOne(dataGrown10);
    %     dataGrown11(dataToMask==0)=0;
    %
    %     dataGrown12 = growByOne(dataGrown11);
    %     dataGrown12(dataToMask==0)=0;
    %
    %     dataGrown13 = growByOne(dataGrown12);
    %     dataGrown13(dataToMask==0)=0;
    %
    %     dataGrown14 = growByOne(dataGrown13);
    %     dataGrown14(dataToMask==0)=0;
    %     clear dataGrown5 dataGrown6 dataGrown7 dataGrown8;
    
    %% Get colormap values from this
    
    %Epi
    %dataGrown14smooth = imgaussfilt3(uint8(dataGrown14),3);
    if mode == 4
        
        % mask with widest region
        
        % dataGrown15 = growByOne(dataGrown14);
        surfaceColormap = AWTcolormap;
        surfaceColormap(dataGrownNew ==0)=0;
        
    elseif mode == 2 || mode == 3
        %Endo
        
        % mask with widest region
        
        
        surfaceColormap = AWTcolormap;
        surfaceColormap(dataGrownNew ==0)=0;
    end
    
    % plot figure
    
    
    % New method which can be used for all surfaces
    % dialatedAWT = zeros(Ny, Nx, Nz);
    %
    %     radius = 3;
    %     [xgrid, ygrid, zgrid] = meshgrid(-radius:radius);
    %     ball = (sqrt(xgrid.^2 + ygrid.^2 + zgrid.^2) <= radius);
    %     dialatedAWT = imdilate(AWTfinal,ball);
    %
    %
    %     dialatedAWT = uint8(dialatedAWT*255/max(max(max(dialatedAWT))));
    %
    %     save AWT_PDE_dialated_colormap_all_surfaces.mat dialatedAWT
    
    %% Remove the first and last layer and define this as the surface, the
    % colormap is now 1 pixel wider on each side compared to the surface
    
    disp('Removing first layers')
    %Epi
    if mode == 1
        
        % mask with 1 pixel narrower
        surfaceToMap = logical(dataGrownOld);
        dataGrown3(dataToMask==0)=0;
        %dataGrown3 = imgaussfilt3(uint8(dataGrown3),3);
        surfaceToMap(dataGrown3>0)=0;
        figure, imagesc(surfaceToMap(:,:,round(Nz/2)));
    elseif mode == 2 || mode == 3
        %Endo
        
        % mask with 1 pixel narrower
        surfaceToMap = logical(dataGrownOld);
        dataGrown2(dataToMask ==0)=0;
        surfaceToMap(dataGrown2>0)=0;
        figure, imagesc(surfaceToMap(:,:,round(Nz/2)));
    end
    
    
    %%
    
    % %% Correct epi surface to fit
    %
    % load Epi_Surface_unmasked.mat
    %
    % newEpi = growByOne(tempCleanFilled);
    % newEpi2 = growByOne(newEpi);
    % newEpi3 = growByOne(newEpi2);
    % newEpi3(epi_AWT == 0)=0;
    %
    % % Write data
    % dataToWrite = newEpi3;
    %
    % [~,~,Nz]=size(dataToWrite);
    %
    % for i=1:Nz
    %     imwrite(dataToWrite(:,:,i),sprintf('H66_epi_surface_22Jan_%03d.tif',i));
    % end
    

    %endoLA_AWT_Surface(maskAppendage>0)=0;

    %endoRA_AWT_Surface(maskAppendage>0)=0;

    %epi_AWT_Surface(maskAppendage>0)=0;
    

    %% save data
    
    
    cd([Heart, '/']);
    
        if mode == 2
            endoLA_AWT_Surface = surfaceToMap;
            endoLA_AWT_Colormap = surfaceColormap;
            disp('Removing appendages from colormaps');
%             endoLA_AWT_Colormap(maskAppendage>0)=0;
            save('WallThickness_Surfaces_and_Colormaps_PDE.mat' , 'endoLA_AWT_Colormap', 'endoLA_AWT_Surface', '-append');
    
        elseif mode == 3
            endoRA_AWT_Surface = surfaceToMap;
            endoRA_AWT_Colormap = surfaceColormap;
            disp('Removing appendages from colormaps');
%             endoRA_AWT_Colormap(maskAppendage>0)=0;
            save('WallThickness_Surfaces_and_Colormaps_PDE.mat', 'endoRA_AWT_Colormap', 'endoRA_AWT_Surface', '-append');
    
        elseif mode == 1
            epi_AWT_Surface = surfaceToMap;
%             epi_AWT_Surface(maskAppendage>0)=0;
            save('WallThickness_Surfaces_and_Colormaps_PDE.mat', 'epi_AWT_Surface');
        elseif mode == 4
            epi_AWT_Colormap = surfaceColormap;
            disp('Removing appendages from colormaps');
%             epi_AWT_Colormap(maskAppendage>0)=0;
            save('WallThickness_Surfaces_and_Colormaps_PDE.mat', 'epi_AWT_Colormap', '-append');
    
        end
    

end

        figure, imagesc(epi_AWT_Colormap(:,:,round(Nz/2)));
        figure, imagesc(endoRA_AWT_Colormap(:,:,round(Nz/2)));
        figure, imagesc(endoLA_AWT_Colormap(:,:,round(Nz/2)));
    
%% Remove appendages
% load WallThickness_Surfaces_and_Colormaps_23Jan.mat
%
% path = './H66_Appendage_Mask_24Jan/';
% files = dir([path '*.tif']);
%
% %Get the dimensions needed for the loops by checking the first file
% Nz = length(files);
% maskAppendage(:,:,1) = imread([path files(1).name]);
% [Nx,Ny] = size(maskAppendage(:,:,1));
% %Read the rest of the images
% for i=2:Nz
%     maskAppendage(:,:,i) = imread([path files(i).name]);
% end



%% Write data
colormapandsurface = [Heart, '_Surfaces_and_Colormaps'];
mkdir(colormapandsurface);
cd(colormapandsurface);


disp('Writing endo LA data');

dataToWrite = endoLA_AWT_Surface;

[~,~,Nz]=size(dataToWrite);

outputPath = [Heart, '_', 'endoLA_surface']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = [outputPath, '_', date];
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

dataToWrite = endoLA_AWT_Colormap;

outputPath = [Heart, '_', 'endoLA_colormap']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = [outputPath, '_', date];
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end
%%
disp('Writing endo RA data');
dataToWrite = endoRA_AWT_Surface;

[~,~,Nz]=size(dataToWrite);

outputPath = [Heart, '_', 'endoRA_surface']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = [outputPath, '_', date];
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

dataToWrite = endoRA_AWT_Colormap;

outputPath = [Heart, '_', 'endoRA_colormap']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = [outputPath, '_', date];
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end
%%
disp('Writing tempCleanFilled data');
% epi_AWT_Surface(maskAppendage>0)=0;
dataToWrite = epi_AWT_Surface;

[~,~,Nz]=size(dataToWrite);

outputPath = [Heart, '_', 'tempCleanFilled_surface']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = [outputPath, '_', date];
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

dataToWrite = epi_AWT_Colormap;

outputPath = [Heart, '_', 'epi_colormap']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = [outputPath, '_', date];
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end
disp('saving AWT surfaces and colormaps');

cd ..
save WallThickness_Surfaces_and_Colormaps.mat endoLA_AWT_Colormap endoLA_AWT_Surface endoRA_AWT_Colormap endoRA_AWT_Surface epi_AWT_Colormap epi_AWT_Surface


% disp('getting appendages only')
% load BinaryAtriaOnly.mat
% load appendageMask.mat
% 
% path = ['./', Heart, '_Cleaned_8bit/'];
% files = dir([path '*.tif']);
% 
% % ADD MANUALY CHANGE FOR NORMAL RUNNING
% Get the dimensions needed for the loops by checking the first file
% Nz = length(files);
% cleaned8bit(:,:,1) = imread([path files(1).name]);
% [Nx,Ny] = size(cleaned8bit(:,:,1));
% Read the rest of the images
% for i=2:Nz
%     cleaned8bit(:,:,i) = imread([path files(i).name]);
% end
% %
% appendages = uint8(zeros(size(cleaned8bit)));
% appendages(maskAppendage > 0 &  cleaned8bit > 0) = 1; % cleaned8bit -> tempCleanFilled
% 
% 
% disp('Writing appendages data');
% dataToWrite = appendages;
% 
% [~,~,Nz]=size(dataToWrite);
% 
% outputPath = [Heart, '_', 'appendages']; % <- Change the last string if wanting to use another output folder name
% mkdir(outputPath);
% fileName = [outputPath, '_', date];
% for i=1:Nz
%     imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
% end

disp('AWT Complete. No More Wall Thickness Tasks');