% clearvars -except Heart
parent = cd;
%%
cd(Heart)

path = './interp_AWT/';
files = dir([path '*.tif']);

%Get the dimensions needed for the loops by checking the first file
Nz = length(files);
input_mask(:,:,1) = imread([path files(1).name]);
[Nx,Ny] = size(input_mask(:,:,1));
%Read the rest of the images
for i=2:Nz
    input_mask(:,:,i) = imread([path files(i).name]);
end

temp = zeros(Nx,Ny,Nz + 20);
temp(:,:,10:Nz+9) = input_mask;
input_mask = temp;
clear temp
cd(parent)

input_mask = Resample_new(input_mask);
[Nxo,Nyo,Nzo] = size(input_mask);
[input_mask,index_array] = crop_region(input_mask);
input_mask = uint8(input_mask);

Lawall = input_mask;
Lawall(input_mask ~= 2) = 0;

Lacavity = input_mask;
Lacavity(input_mask ~=4) = 0;

Rawall = input_mask;
Rawall(input_mask ~=1) = 0;

Racavity = input_mask;
Racavity(input_mask ~=3) = 0;


Raclosing = Racavity;
for i = 1:2*3
    Raclosing = growByOne_2D(Raclosing); 
%     if i == 1
%         Raendo = Raclosing;
%         Raendo(Racavity > 0) = 0;
%     end

end
Raclosing(Racavity > 0) = 0;

Laclosing = Lacavity;
for i = 1:2*3
    Laclosing = growByOne_2D(Laclosing);  
%     if i == 1
%         Laendo = Laclosing;
%         Laendo(Lacavity > 0) = 0;
%     end
end
Laclosing(Lacavity > 0) = 0;

% 
% temp = Lawall(startX-padding:finishX+padding,startY-padding:finishY+padding,startZ-padding:finishZ+padding);
% clear Lawall
% Lawall = temp;
% clear temp
% 
% temp = Lacavity(startX-padding:finishX+padding,startY-padding:finishY+padding,startZ-padding:finishZ+padding);
% clear Lacavity
% Lacavity = temp;
% clear temp
% 
% temp = Rawall(startX-padding:finishX+padding,startY-padding:finishY+padding,startZ-padding:finishZ+padding);
% clear Rawall
% Rawall = temp;
% clear temp
% 
% temp = Racavity(startX-padding:finishX+padding,startY-padding:finishY+padding,startZ-padding:finishZ+padding);
% clear Racavity
% Racavity = temp;
% clear temp
% 
% temp = Raclosing(startX-padding:finishX+padding,startY-padding:finishY+padding,startZ-padding:finishZ+padding);
% clear Raclosing
% Raclosing = temp;
% clear temp
% 
% temp = Laclosing(startX-padding:finishX+padding,startY-padding:finishY+padding,startZ-padding:finishZ+padding);
% clear Laclosing
% Laclosing = temp;
% clear temp

% Laepi = Laclosing;
% Laepi = growByOne_2D(Laepi);
% Laepi(Laclosing ==0) = 0;
% 
% Raepi = Raclosing;
% Raepi = growByOne_2D(Raepi);
% Raepi(Raclosing ==0) = 0;

mask = Raclosing + Laclosing;
mask(Lawall > 0) = 0;
mask(Rawall > 0) = 0;

cd(Heart)
save atriaClosingMask.mat mask


dataToWrite = mask;
mkdir('Atria_Closing_Mask')
cd('Atria_Closing_Mask')

[~,~,Nz]=size(dataToWrite);

for i=1:Nz
    imwrite(dataToWrite(:,:,i),sprintf('Atria_Closing_Mask_%03d.tif',i));
end
cd ..

closedAtriaOutput = logical(Raclosing + Laclosing + Lawall + Rawall);
save closedAtriaOutput.mat closedAtriaOutput

tempCleanFilled = logical(Lawall + Rawall);
save BinaryAtriaOnly.mat tempCleanFilled

endoLACavity = Lacavity;
save 3DCavityLAendo.mat endoLACavity

endoRACavity = Racavity;
save 3DCavityRAendo.mat endoRACavity

save indexs.mat index_array Nxo Nyo Nzo

[Nx,Ny,Nz] = size(Raclosing);
save parameters.mat Nx Ny Nz





