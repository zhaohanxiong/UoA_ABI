% 11
%%
%Create Gradient Field
% Author: Aaqel Nalar (Jan 2019)
%% Load data
parent = pwd;
cd(parent)
cd([Heart, '/']);

tic
disp('Reading data');
%load EndoToEndoLaplaceFieldLines.mat
load TransmuralFieldLines.mat % From LaplaceSolverwithMiddleLine (10)
load parameters.mat % From Noise Removal (1)
swap = Nx;
Nx = Ny;
Ny = swap;
clear swap
%% find gradient field

disp('Calculating gradient field');
[px, py, pz] = gradient(tissue, eps);
% gradient field
dtissue(:,:,:,1) = py; %y direction gradient (the fact that its named wrong doesnt matter as it is reconstructed the same way)
dtissue(:,:,:,2) = px; %x direction gradient (the fact that its named wrong doesnt matter as it is reconstructed the same way)
dtissue(:,:,:,3) = pz; %z direction gradient

% figure, histogram(nonzeros(px));
% figure, histogram(nonzeros(px));
% figure, histogram(nonzeros(px));


% normalise gradient field

disp('Normalising gradient field');
Ntissue = zeros(Ny, Nx, Nz,3);
l = sqrt(dtissue(:,:,:,1).^2 + dtissue(:,:,:,2).^2 + dtissue(:,:,:,3).^2);
l(l==0) = eps; 
 
Ntissue(:,:,:,1) = dtissue(:,:,:,1)./l;
Ntissue(:,:,:,2) = dtissue(:,:,:,2)./l;
Ntissue(:,:,:,3) = dtissue(:,:,:,3)./l;

clear dtissue px py pz
%% save
disp('Saving gradient field');
gradientFieldX = Ntissue(:,:,:,1);
gradientFieldY = Ntissue(:,:,:,2);
gradientFieldZ = Ntissue(:,:,:,3);
save -v7.3 gradientField.mat gradientFieldX gradientFieldY gradientFieldZ

toc

disp('Complete');
disp('Run AWT PDE method');