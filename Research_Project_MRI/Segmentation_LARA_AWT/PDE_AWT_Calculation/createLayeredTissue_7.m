%7
%%
%Create layered information
%Obtain inside tissue
%Gives label values to each of the tissue layers.
% Also removes artificial layers of mask from the surface masks 

% Author: Aaqel Nalar (Jan 2019)

%% Read tissue data
parent = pwd;
%date = input('Enter the date as a string including apostrophes (e.g. ''11Feb''): ');
cd(parent)
cd([Heart, '/']);

tic 
% load parameters.mat % From Noise Removal (1)
disp('loading data');
load closedAtriaOutput.mat %From Process Closed Atria (4)
load 3DCavityLAendo.mat %From regionGrowing Processing (5)
load 3DCavityRAendo.mat %From regionGrowing Processing (5)
load endoLA_surface_masked.mat %From createSurfaceMasks_New (6b)
load endoRA_surface_masked.mat %From createSurfaceMasks_New (6b)
load epi_surface_masked.mat %From createSurfaceMasks_New (6b)
[Nx,Ny,Nz] = size(closedAtriaOutput);

% closeAtriaMask = zeros(Nx,Ny,Nz,'uint8');
% maskPath = './H109_AtriaClosingMask_11Feb/';
% maskFiles = dir([maskPath '*.tif']);
% for i=1:Nz
%     closeAtriaMask(:,:,i) = imread([maskPath maskFiles(i).name]);
% end

%% Label all data

disp('labelling data')
data = zeros(Nx,Ny,Nz,'uint8');
outside = data;

data(closedAtriaOutput >= 1) = 1; 

data(endoRACavity > 0) = 2;

data(endoLACavity > 0) = 3;

data(epi_surface > 0) = 4;

data(endoRA_surface > 0) = 5;

data(endoLA_surface > 0) = 6;

outside(data == 0) = 1;

%% Find internal Holes
Labels = bwlabeln(outside,6);
stats = regionprops(Labels,'Area');
CRegions = [stats.Area];
[~,biggest] = max(CRegions);
dataTemp = outside;
dataTemp(Labels==biggest) = 0;

disp('Removing enclosed holes automatically');
internalHoles = dataTemp;

%% Assign internal holes to be tissue inside
data(internalHoles > 0) = 1;

%% Remainder is outside.
outside = zeros(Nx,Ny,Nz,'uint8');
outside(data == 0) = 1;

save outside.mat outside
%% Extract Tissue Inside
disp('creating and saving tissue inside');

tissueInside = data;
tissueInside(data ~= 1) = 0;
save tissueInside.mat tissueInside;
%% Save data
disp('saving and writing labelled data');

labelledData = data;

save labelledData.mat labelledData


%% Write data

dataToWrite = labelledData;
outputPath = [Heart, '_', 'Labelled_Data']; % <- Change the last string if wanting to use another output folder name

mkdir(outputPath);

fileName = outputPath;
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end


%% Show data

% disp('Showing a slice')
% figure
% imagesc(labelledData(:,:,round(Nz/2)));
% figure
% imagesc(outside(:,:,round(Nz/2)));
toc

disp('Complete');
disp('Run First Laplace Solver (8) Script');