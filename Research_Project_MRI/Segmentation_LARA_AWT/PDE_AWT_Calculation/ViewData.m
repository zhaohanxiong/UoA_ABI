%% Normal TIF stack
dataToShow = tissue;
pauseTime = true;


[Nx,Ny,Nz]=size(dataToShow);

figure,
for i=1:Nz
    h = imagesc(squeeze(dataToShow(:,:,i)));
    colorbar();
     colormap(jet)
%    caxis([100 300])
    %h = imshow(dataToShow(:,:,i));
    if(pauseTime)
        pause(0.0167);
        i
    end
end


%% Colour fields

dataToShow2 = newmid;

figure, imagesc(dataToShow2(:,:,222)), colorbar%colormap(jet);
figure, imagesc(dataToShow2(:,:,102)), colormap(jet);
figure, imagesc(dataToShow2(:,:,103)), colormap(jet);
figure, imagesc(dataToShow2(:,:,104)), colormap(jet);
figure, imagesc(dataToShow2(:,:,97)), colormap(jet);
%% Write data
dataToWrite = uint16(tempCleanFilled);
% E(:,:,:,69)
[~,~,Nz]=size(dataToWrite);

for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf('clean%03d.tif',i));
end

%% Read data
% cd '/hpc_atog/anl484/Jichao/ABI'
% cd '/hpc_htom/jken335/ExVivo_James'

path = './H145_Frontwall/';
files = dir([path '*.tif']);

%Get the dimensions needed for the loops by checking the first file
Nz = length(files);
re(:,:,1) = imread([path files(1).name]);
[Nx,Ny] = size(re(:,:,1));
%Read the rest of the images
for i=2:Nz
    re(:,:,i) = imread([path files(i).name]);
end


% cd '/hpc_atog/anl484/Jichao/ExVivoAnalysis'
% 
% save MRIClean.mat MRIClea
%% 