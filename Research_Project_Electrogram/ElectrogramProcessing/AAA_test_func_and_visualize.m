%% load some data
clearvars; close all;
load('C:\Users\zxio506\Desktop\Vm_2stable_rotor_large_150.mat')
%Vm = Vm(:,:,:,300:1300);

%imagesc(squeeze(Vm(:,:,3,50)));
for i = 1:size(Vm,4)
    imagesc(squeeze(Vm(:,:,3,i)));
    pause(0.1);fprintf('%d\n',i);
end


x = 120;
y = 140;
for i = 1:size(Vm,4)
    imagesc(squeeze(Vm(x:(x+240),y:(y+240),3,i)));
    pause(0.01);fprintf('%d\n',i);
end

%% transmembrane to extra cellular potential
[Phi] = TransMembrane2ExtraCellularPotential(Vm,40);

% save('C:\Users\zxio506\Desktop\Phi_drifting_rotor_50x50_J_fi x 7.mat','phi')
% nfiles = length(dir('C:/Users/zxio506/Desktop/Test/pp*'));
% Phi = zeros([8 8 nfiles],'single');
% for j = 1:1:nfiles
%     infile = fullfile('C:/Users/zxio506/Desktop/Test', sprintf('pp%04d.vtk',j));
%     fid = fopen(infile,'r');
%     aa = fscanf(fid, '%g', [1 inf]);
%     fclose(fid);
%     counter0 = 0;
%     for Yi=1:1:8
%         for Xi=1:1:8
%             counter0 = counter0 + 1;
%             Phi(Yi, Xi, j) = aa(counter0);
%         end
%     end
% end;

% pos = 3;
% figure, plot(squeeze(Phi(pos,pos,:)));
% figure, plot(squeeze(Vm(int8(pos*240/8),int8(pos*240/8),3,:)));
% 
for i = 1:size(Phi,3)
    imagesc(Phi(:,:,i));colormap('Jet');
    pause(0.025);fprintf('%d\n',i);
end

%% unipolar to phase map
%load('C:\Users\zxio506\Desktop\Phi_0.025_160_1.0.mat')
[Phase_Map] = ExtraCellularUnipolarPotential2PhaseMap(double(Phi));

% figure, imagesc(squeeze(Phase_Map(:,:,end/2)));colormap('Jet');
for i = 1:size(Phase_Map,3)
    imagesc(Phase_Map(:,:,i));colormap('Jet');
    pause(0.025);fprintf('%d\n',i);
end

%% phase singularity tracing
[PS_trajectory] = LocalizePhaseMapSingularity(Phase_Map);

for i = 1:(size(Phase_Map,3)-30)
    
    PS = PS_trajectory{i};
    
    subplot(1,2,1);
    imagesc(Phase_Map(:,:,i));colormap(gca,'Jet')
    hold on;
    scatter(PS(:,1),PS(:,2),200,'MarkerEdgeColor','w','MarkerFaceColor','k','linewidth',2.5);
    
    subplot(1,2,2);
    plot(PS(:,1),PS(:,2),'b-o');
    axis([0 size(Phase_Map,1) 0 size(Phase_Map,2)]);
    set(gca, 'YDir','reverse');
    
    pause(0.1);fprintf('%d\n',i);
    
end 