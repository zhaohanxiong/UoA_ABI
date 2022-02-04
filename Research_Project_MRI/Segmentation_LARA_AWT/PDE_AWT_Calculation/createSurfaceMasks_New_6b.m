%6B
%%
parent = pwd;
%date = input('Enter the date as a string including apostrophes (e.g. ''11Feb''): ');
cd(parent)
cd([Heart, '/']);
%create surface layer
%creates a layer outside the cavities to represent the tissue of the LA, RA
%endo and epicardial tissue

% NOTE: Create write file locations first

%Author: Aaqel Nalar, Dec 2018

%% Read cavities and atria closing mask
tic
view = 0;

load parameters.mat %From Noise Removal (1)

disp('Reading LA cavity');
load 3DCavityLAendo.mat  %From regionGrowing Processing (5)

disp('Reading RA cavity');
load 3DCavityRAendo.mat %From regionGrowing Processing (5)

disp('Reading closed atria output');
load closedAtriaOutput.mat %From ProcessClosedAtria (4)

load BinaryAtriaOnly.mat %From Morphological Operations (3)

disp('Reading Closing Mask: MAKE SURE THIS IS CORRECT IN THE CODE');

%change accordingly
files = dir([Aclosing '*.tif']);

%Read the rest of the images
for i=1:Nz
    ClosedAtriaMask(:,:,i) = imread([Aclosing files(i).name]);
end

% %% Read mask data - currently unused beacause no effect
% mask = zeros(Nx,Ny,Nz, 'uint8');
% maskPath = './H66_Atria_Closing_mask2_4Dec/';
% maskFiles = dir([maskPath '*.tif']);
% for i=1:Nz
%     mask(:,:,i) = imread([maskPath maskFiles(i).name]);
% end
%
% surface(mask ~= 0) = 0;

%% Process cavities and obtain surfaces
cd ..
disp('Processing LA surface');

temp = growByOne_2D(endoLACavity);
endoLA_surface = growByOne_2D(temp);
endoLA_surface(endoLACavity > 0) = 0;
%%
disp('Processing RA surface');

temp = growByOne_2D(endoRACavity);
endoRA_surface = growByOne_2D(temp);
endoRA_surface(endoRACavity > 0) = 0;
%%
disp('Processing Epicardial');

%enlarge closed atria bhy 2 pixels
temp = growByOne_2D(closedAtriaOutput);
line = growByOne_2D(temp);
%  remove original to get hollow shell
line(closedAtriaOutput > 0) = 0;
line(endoRACavity >0) = 0;
line(endoLACavity > 0) = 0;
% remove inner materials
disp('Removing inner regions');
    Labels = bwlabeln(line,6);
    stats = regionprops(Labels,'Area');
    CRegions = [stats.Area];
    [~,biggest] = max(CRegions);
    finalLine = line;
    finalLine(Labels~=biggest) = 0;
% grow hollow shell to be 2 pixels inwards
temp = growByOne_2D(finalLine);
epi_surface = growByOne_2D(temp);
% Retain only what was in the original tissue and remove extra layers
epi_surface(closedAtriaOutput == 0) = 0;



%%
% use LA and RA cavity and closed atria to reverse solve for epicardial
%interior is the sum of all 3 cavities
cd([Heart, '/']);
disp('Saving sealed versions');
save endoLA_surface_masked.mat endoLA_surface
save endoRA_surface_masked.mat endoRA_surface
save epi_surface_masked.mat epi_surface

%% Unmask
disp('Unmasking surfaces');
cd(parent);

% Grow size of mask to coincide with surfaces
maskTemp = ClosedAtriaMask;
maskTemp(tempCleanFilled > 0) = 0;
mask2 = growByOne_2D(maskTemp);
mask3 = growByOne_2D(mask2);

% Remove hole fill structures
endoRA = endoRA_surface;
% 2 is the value of the artificial holes (not structural fills)
% e.g. SVC hole filling
endoRA(mask3 == 2) = 0;

endoLA = endoLA_surface;
endoLA(mask3 == 2) = 0;

epicardial = epi_surface;
epicardial(mask3 == 2) = 0;

cd([Heart, '/']);

% if view == 1
%     figure
%     dataShow = epicardial.*255;
%     for i=1:Nz
%         h = imshow(dataShow(:,:,i));
%         % pause(0.1);
%     end
% end

disp('saving data');
save endoRA_surface.mat endoRA
save endoLA_surface.mat endoLA

disp('saving epi surface');
save epi_surface.mat epicardial
%% write data (move to desired location first)

disp('LA surface writing to file');
dataToWrite = endoLA;
outputPath = [Heart, '_', 'endoLA_surface']; % <- Change the last string if wanting to use another output folder name

mkdir(outputPath);

fileName = outputPath;
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

disp('RA surface writing to file');
dataToWrite = endoRA;
outputPath = [Heart, '_', 'endoRA_surface']; % <- Change the last string if wanting to use another output folder name

mkdir(outputPath);

fileName = outputPath;
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

disp('epicardial surface writing to file');
dataToWrite = epicardial;
outputPath = [Heart, '_', 'epi_surface']; % <- Change the last string if wanting to use another output folder name

mkdir(outputPath);

fileName = outputPath;
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

disp('Finished Writing');
disp('Complete');
disp('Run Create layered tissue (7) Script');
toc