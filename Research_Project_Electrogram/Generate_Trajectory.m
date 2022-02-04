%% Save all trajectories
clearvars;
addpath('Visualize_PS');

% list all file names
data_path = 'C:\Users\zxio506\Desktop\AP';
files = ls(data_path);
files = files(3:end,:);

for n = 1:length(files)
    
    disp(n)
    % compute trajectory
    load(fullfile(data_path,files(n,:))); %load('C:\Users\zxio506\Desktop\Vm_cAF.mat');
    Vm_temp = single(zeros([size(Vm,1),size(Vm,2),3,size(Vm,4)]));
    for i = 1:3
        Vm_temp(:,:,i,:) = Vm;
    end
    tt = 3;
    [Phi_actual] = TransMembrane2ExtraCellularPotential(Vm_temp(:,:,:,1:tt:end),60);
    [Xq,Yq,Zq] = meshgrid(linspace(1,60,60),linspace(1,60,60),linspace(1,size(Phi_actual,3),size(Phi_actual,3)*tt));
    %Phi = interp3(Phi_actual,Xq,Yq,Zq,'linear');
    %[Xq,Yq,Zq] = meshgrid(linspace(1,60,240),linspace(1,60,240),linspace(1,size(Phi,3),size(Phi,3)));
    Phi = interp3(Phi_actual,Xq,Yq,Zq,'spline');
    save(fullfile(data_path,strcat('Phi_',files(n,:))),'Phi'); %save('C:\Users\zxio506\Desktop\Phi_cAF.mat','Phi');
	
    [Phase_Map] = ExtraCellularUnipolarPotential2PhaseMap(double(Phi),1000);
    [PS_trajectory] = LocalizePhaseMapSingularity(Phase_Map);
    save(fullfile(data_path,strcat('PS_trajectory_',files(n,:))),'PS_trajectory'); %save('C:\Users\zxio506\Desktop\PS_trajectory_cAF.mat','PS_trajectory');
    
    % make density plot
    PS_num = cellfun('length',PS_trajectory);
    temp_lab = zeros(length(PS_num),1);
    for i = 1:length(PS_num)-10
		temp_lab(i,1) = sum(PS_num(i:i+10) < 3) == 10;
    end

	density_plot = zeros([60,60]);
	PS_line = [];
    for i = find(temp_lab==1,1):find(temp_lab==1,1,'last')
        PS = round(PS_trajectory{i});
        if ~(isnan(PS(end,1)) || isnan(PS(end,2)))
            density_plot(PS(end,1),PS(end,2)) = density_plot(PS(1,1),PS(1,2)) + 1;
        end
		PS_line = [PS_line; PS(end,:)];
    end
	
    % occurances in sub-region
    n_window = 8/4;
    density_subregion = density_plot;
    max_subregion = density_plot;
    
    for i = 1:n_window:size(density_plot,1)-n_window
        for j = 1:n_window:size(density_plot,2)-n_window
            density_subregion(i:(i+n_window),j:(j+n_window)) = sum(sum(density_plot(i:(i+n_window),j:(j+n_window))));
            max_subregion(i:(i+n_window),j:(j+n_window)) = max(max(density_plot(i:(i+n_window),j:(j+n_window))));
        end
    end
    
    % display and save plot
    subplot(2,2,1); imagesc(density_plot');colormap(gca,'Jet');colorbar;title('Density')
    subplot(2,2,2); plot(PS_line(1:end-10,1),PS_line(1:end-10,2),'b-');axis([0 60 0 60]);set(gca,'Ydir','reverse');colorbar;title('Trajectory')
    subplot(2,2,3); imagesc(density_subregion');colormap(gca,'Jet');colorbar;title('Density')
    subplot(2,2,4); imagesc(max_subregion');colormap(gca,'Jet');colorbar;title('Max Value')
    %imagesc(Phi(:,:,1));colormap('jet');colorbar;caxis([-100 100])
    %imagesc(max(max(density_subregion))-density_subregion');colormap(gca,'Hot');colorbar;title('Density')
    %imagesc(0-Phi(:,:,200));colormap('gray');colorbar;caxis([-100 100])
    fig = gcf;fig.PaperPosition = [0 0 10 8];
    saveas(fig,sprintf('singularities\\Density_Plot_%s.png',files(n,:)));
    
end



%% Visualization of trajectories
% PS_line = [];
% 
% for i = 1:length(PS_trajectory)
%     
%     PS = PS_trajectory{i};
%     PS_line = [PS_line; PS(1,:)];
%     
%     subplot(1,2,1);
%     imagesc(Phase_Map(:,:,i));colormap(gca,'Jet')
%     hold on;
%     scatter(PS(:,1),PS(:,2),200,'MarkerEdgeColor','w','MarkerFaceColor','k','linewidth',2.5);
%     
%     subplot(1,2,2);
%     plot(PS_line(:,1),PS_line(:,2),'b-');
%     %plot(PS(:,1),PS(:,2),'bo');
%     axis([0 size(Phase_Map,1) 0 size(Phase_Map,2)]);
%     set(gca, 'YDir','reverse');
%     
%     pause(0.01);fprintf('%d\n',i);
%     
% end

%% Check Predictions
i = 10;

%load(sprintf('C:\\Users\\zxio506\\Desktop\\model\\temp_pred%i.mat',i))

load('C:\\Users\\zxio506\\Desktop\\germany_pred_label.mat')

label = true;

x1 = 12;
x2 = 22;
y1 = 15;
y2 = 25;

thres_pred = 0.20;
thres_true = 0.19;

%smooth prediction region
%temp_lab = squeeze(test_label(i,:,:));

% dilate pred
%temp_pre = pred;

%pred(pred < max(max(pred))*thres_pred) = 0;
%pred(pred >= max(max(pred))*thres_pred) = 1;
%label(label < max(max(label))*thres_true) = 0;
%label(label >= max(max(label))*thres_true) = 1;

subplot(1,2,1)
imagesc(label(x1:x2,y1:y2));colorbar;colormap('jet');
subplot(1,2,2)
imagesc(pred(x1:x2,y1:y2));colorbar;colormap('jet');

%% get centroid evaluation value
% set threshold
label = true;

thres_pred = 0.16;
thres_true = 0.13;

% apply thresholds
pred(pred < max(max(pred))*thres_pred) = 0;
pred(pred >= max(max(pred))*thres_pred) = 1;
%label(label < max(max(label))*thres_true) = 0;
%label(label >= max(max(label))*thres_true) = 1;

% find centroid location
stats = regionprops(pred);
pred_centroid = stats.Centroid;
stats = regionprops(label);
label_centroid = stats.Centroid;

% find MSE
MSE = sum(sqrt((pred_centroid - label_centroid).^2))/30*120