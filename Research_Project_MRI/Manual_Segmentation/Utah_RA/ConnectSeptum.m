%% Manual coorection to Connect Septum and remove excess material
% Author Aaqel Nalar - Mar 2019
% Make sure cavity dialation is run


%% MAKE THE LABELS IN AMIRA FOR INPUT AS SUCH:

% Exterior (Default)- 0
% Inside (Default) -1
% Septum - 2
% Remove from RA wall - 3
% Add to RA wall - 4

%%
clear

%path = './CARMA1160/pre/CARMA_1160_pre_NewSegmentation/';

heart = input('Enter heart number: ');
state = input('Enter state (e.g. 4mo, pre,) as a string including '''': ');
tic

cd C:/Users/zxio506/Desktop;
path = './NewSegmentation/';
files = dir([path '*.tif']);

%Get the dimensions needed for the loops by checking the first file
Nz = length(files);
data(:,:,1) = imread([path files(1).name]);
[Nx,Ny] = size(data(:,:,1));


%Read the rest of the images
for i=2:Nz
    data(:,:,i) = imread([path files(i).name]);
end


%% mask 

path = './SeptumConnection/';
files = dir([path '*.tif']);

%Get the dimensions needed for the loops by checking the first file
Nz = length(files);
mask(:,:,1) = imread([path files(1).name]);
[Nx,Ny] = size(mask(:,:,1));

%Read the rest of the images
for i=2:Nz
    mask(:,:,i) = imread([path files(i).name]);
end


data(mask == 2) = 5; % Label septum,
% change to data(mask==2) = 1  - to add septum to RA wall


data(mask == 3) = 0; % Excess Cavity and wall Removal
% data(mask == 5) = 1;

%data(mask == 4) = 1; % Add to RA wall


outputPath = ['./CARMA_' num2str(heart),'_', state,'_full'];
mkdir(outputPath);

dataToWrite = uint8(data);

[~,~,Nz]=size(dataToWrite);

for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/CARMA_', num2str(heart),'_',state,'_NewSeg_w_Septum_%03d.tif'],i));
end

%%
toc
dataToShow2 = data;

%figure, imagesc(dataToShow2(:,:,15));