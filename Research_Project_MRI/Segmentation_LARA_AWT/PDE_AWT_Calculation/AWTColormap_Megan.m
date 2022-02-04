%% load AWT final

clear

parent = pwd;
Heart = input('Enter Heart Number/ Folder name as string including apostrophes (e.g. ''H109''): ');
cd(parent)
cd([Heart, '/']);

disp('Make sure data for AWTmap is correct in code: ');
AWTmap = 'Unrounded_AWT_CARMA1421_4mo_results_test2_PDE.mat' % change accordingly
load(AWTmap);
load parameters.mat

% path = './H394_Surface_mask/';
% files = dir([path '*.tif']);
% 
% %Get the dimensions needed for the loops by checking the first file
% Nz = length(files);
% mask(:,:,1) = imread([path files(1).name]);
% [Nx,Ny] = size(mask(:,:,1));
% %Read the rest of the images
% for i=2:Nz
%     mask(:,:,i) = imread([path files(i).name]);
% end


%% Load surface
load BinaryAtriaOnly.mat

%%convert to uint 8 for AWTfinal
AWTfinal(AWTfinal<= 0.625) = 0;
% threshold = quantile(reshape(nonzeros(AWTfinal),[],1),0.90);
% AWTfinal(AWTfinal>= threshold) = threshold;


disp('converting to grayscale');
dataToConvert = AWTfinal;

AWTcolormap = (dataToConvert/max(max(max(AWTfinal))))*255; %% This is an absolute max value so colors can be compared
AWTcolormap = uint8(AWTcolormap);
% AWTcolormap(mask >= 1) = 90;
undialated = AWTcolormap; 

%% grow by one x5 for AWTfinal = colormap
cd(parent)
surfaceData = growByOne(tempCleanFilled);

for i =1:7
    i
    AWTcolormap = growByOne(AWTcolormap);
end
cd(Heart)
%%


dataToWrite = AWTcolormap;

outputPath = [Heart, '_', 'colormap_new']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = outputPath;
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

dataToWrite = surfaceData;

outputPath = [Heart, '_', 'surface_new']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = outputPath;
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

dataToWrite = undialated;

outputPath = [Heart, '_', 'undialated_new']; % <- Change the last string if wanting to use another output folder name
mkdir(outputPath);
fileName = outputPath;
for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/', fileName, '_%03d.tif'],i));
end

