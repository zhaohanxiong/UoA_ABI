%% Loading the data
% this script loads all the data in the "test" folder generated by solving
% the PDEs numerically with the C++ code

clear all; close all; clc

Nx = 240;
Ny = 240;
Nz = 1;

% read the files from the "Test" folder
files = dir('C:/Users/zxio506/Desktop/Test9/a*');
nfiles = length(files);

E = zeros([Ny Nx Nz nfiles], 'single');

for j = 1:1:nfiles
    infile = fullfile('C:/Users/zxio506/Desktop/Test9', sprintf('aa%04d.vtk',j));
    fid = fopen(infile, 'r');
    aa = fscanf(fid, '%g', [1 inf]);
    fclose(fid);
    counter0 = 0;
    for plane=1:1:Nz,
      for Yi=1:1:Ny,
        for Xi=1:1:Nx,
            counter0 = counter0 + 1;
            E(Yi, Xi, plane,j) = aa(counter0);
        end
      end
    end
    disp(j)
end

fprintf('\n\nThe atrial dimension is Ny=%d, Nx=%d, Nz=%d',Ny,Nx,Nz);
fprintf('\nNumber of files loaded (time steps) = %d\n',nfiles);

%% save the data
Vm = single(E);
save('C:\Users\zxio506\Desktop\temp2.mat','Vm','-v7.3');

%% visualise the outpout
close all;

% 1st and 2nd axis is the 2D matrix representing the atrial surface
% 3rd axis is simply for numerical reasons, so always use 3
% 4th axis is the time steps (also the numnber of files output

% plot propagation pattern for atrial surface (2D representation)
%figure, imagesc(squeeze(E(:,:,3,1)'));

% plot the individual signals, note the the 4th axis is the time
%figure, plot(squeeze(E(120,120,3,:)));

%% create a movie
close all;

imagesc(squeeze(E(:,:,1,end)'));
% for i = 1:size(E,4)
%     imagesc(squeeze(E(:,:,1,i)'));
%     pause(0.001)
%     fprintf('%d\n',i);
% end