%% The input to this file needs to be the RA cavity and wall in 1 label
% This is equivalent to all of the white regions associated with the RA
% Author: Aaqel Nalar - Mar 2019
clear
%% MAKE THE LABELS IN AMIRA FOR INPUT AS SUCH:

% Exterior (Default)- 0
% Inside (Default) -1
% LA (everything including wall and cavity)- 2
% RA (everything including wall and cavity)- 3

%% Read amira data outlining cavity and wall for RA (manual) and LA (Utahs)
%path = './CARMA1160/pre/CavityMask/';
heart = input('Enter heart number: ');
state = input('Enter state (e.g. 4mo, pre,) as a string including '''': ');
tic

cd C:/Users/zxio506/Desktop;
path = './CavityMask/';
files = dir([path '*.tif']);

%Get the dimensions needed for the loops by checking the first file
Nz = length(files);
CavityAndWall(:,:,1) = imread([path files(1).name]);
[Nx,Ny] = size(CavityAndWall(:,:,1));


%Read the rest of the images
for i=2:Nz
    CavityAndWall(:,:,i) = imread([path files(i).name]);
end

bothCavities = CavityAndWall;

CavityAndWall(CavityAndWall == 2) = 0; % 2 is the LA, 3 is RA
OriginalRACavityandWall = CavityAndWall; % Only 3 remains

%% Convert to a thin layer and grow to the desired wall thickness size
cd('C:/Users/zxio506/OneDrive/PhD/LARA Segmentation/manual segmentation');

N = 7; % gives a thickness of N-1

for i = 1:N
    CavityAndWall2 = growByOne(CavityAndWall);
    
    if i == 1
        CavityAndWall2(OriginalRACavityandWall>0) = 0; %Now a 1 pixel thick line exists that is 1 layer outside the original cavity
    end
    
    CavityAndWall = CavityAndWall2;
end

RAwall = CavityAndWall; % 2N-1 Pixels thick
RAwall(OriginalRACavityandWall == 0) = 0; % Remove any layers outside the original seal -> N-1 Pixel Thickness Remains
RACavity = uint8(zeros(size(CavityAndWall)));

% what ever was in the original label but not in the RA wall is the cavity
RACavity(OriginalRACavityandWall > 0 & RAwall == 0) = 1; % Obtain Cavity 


%% Use original shape (Erins) and grow it to remove unwanted wall
cd C:/Users/zxio506/Desktop
% used just to speed up the removal in the next step
% 
% path = ['./CARMA', num2str(heart),'/', state, '/Segmentation_', state, '/'];
% files = dir([path '*.tif']);
% 
% %Get the dimensions needed for the loops by checking the first file
% Nz = length(files);
% OriginalShape(:,:,1) = imread([path files(1).name]);
% [Nx,Ny] = size(OriginalShape(:,:,1));
% %Read the rest of the images
% 
% for i=2:Nz
%     OriginalShape(:,:,i) = imread([path files(i).name]);
% end

%%

% % grow erins mask to be very large
% N = 17;
% 
% for i = 1:N
%     OriginalShapeGrown = growByOne(OriginalShape);
%     OriginalShape = OriginalShapeGrown;
% end



finalData = uint8(zeros(size(CavityAndWall)));

finalData(RAwall>0) = 1; % RAW
% for i = 3:13 % select the range where the RA is open (change accordingly)
%     temp = finalData(:,:,i);
%     temp(OriginalShapeGrown(:,:,i) == 0) = 0; % Remove artificial walls from these slides
%     finalData(:,:,i) = temp;
% end
finalData(RACavity>0) = 3; %RAC


%%
%path = './CARMA1160/pre/LAwall/';
path = './LAwall/';
files = dir([path '*.tif']);

%Get the dimensions needed for the loops by checking the first file
Nz = length(files);
LAwall(:,:,1) = imread([path files(1).name]);
[Nx,Ny] = size(LAwall(:,:,1));
%Read the rest of the images

for i=2:Nz
    LAwall(:,:,i) = imread([path files(i).name]);
end
% 
% path = ['./CARMA', num2str(heart),'/', state, '/Trimmask/'];
% files = dir([path '*.tif']);
% 
% %Get the dimensions needed for the loops by checking the first file
% Nz = length(files);
% TrimMask(:,:,1) = imread([path files(1).name]);
% [Nx,Ny] = size(TrimMask(:,:,1));
% %Read the rest of the images
% 
% for i=2:Nz
%     TrimMask(:,:,i) = imread([path files(i).name]);
% end
% 
% LAwall(TrimMask>0) = 0;

finalData(LAwall>0) = 2; % LAW


%path = './CARMA1160/pre/LAendoNoVeins/';
path = './LAendoNoVeins/';
files = dir([path '*.tif']);

%Get the dimensions needed for the loops by checking the first file
Nz = length(files);
LACavity(:,:,1) = imread([path files(1).name]);
[Nx,Ny] = size(LACavity(:,:,1));
%Read the rest of the images

for i=2:Nz
    LACavity(:,:,i) = imread([path files(i).name]);
end

% LACavity(TrimMask>0) = 0;

finalData(LACavity>0) = 4; % LAC


%%

%outputPath = './CARMA1160/pre/CARMA_1160_pre_NewSegmentation';
outputPath = './NewSegmentation';
mkdir(outputPath);

dataToWrite = finalData;

[~,~,Nz]=size(dataToWrite);

for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf([outputPath, '/CARMA_', num2str(heart),'_',state,'_NewSeg_%03d.tif'],i));
end

%%

toc
dataToShow2 = finalData;

%figure, imagesc(dataToShow2(:,:,15));