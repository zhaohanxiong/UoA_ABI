%% Initialization
clearvars;
mkdir 'C:\Users\zxio506\Desktop\ECP';

data_ls = {'Vm_stable_rotor_large.mat', ...
		   'Vm_drifting_rotor_large_0.01_small_drift.mat', ...
		   'Vm_drifting_rotor_large_0.01_0.05_breakup_drift.mat', ...
		   'Vm_drifting_rotor_large_big_drift.mat', ...
		   'Vm_2stable_rotor_large_40.mat', ...
		   'Vm_2stable_rotor_large_150.mat'};
       
sample_coords = {%y   x
				[90,135 ; 170,210 ; 20,210 ; 170,50 ; 20,50], ... %[175,140],[80,220],[10,140],[80,50],
				[90,135 ; 170,210 ; 40,210 ; 180,50 ; 20,60], ...
				[80,130 ; 150,200 ; 40,210 ; 160,75 ; 40,90], ...
				[120,120; 200,180 ; 50,180 ; 50,30 ; 40,90], ...
				[80,140 ; 10,140 ; 160,140], ...
				[80,140 ; 10,140 ; 180,140]
				};

lim1 = [1,80,220,250,350,350];

%% Process Data
for i = 1:length(data_ls)
    
    load(fullfile('C:\Users\zxio506\Desktop',data_ls{i}))
    coord = sample_coords{i};
    
    for j = 1:size(coord,1)

        [Phi_actual] = TransMembrane2ExtraCellularPotential( ...
            Vm(coord(j,1):(coord(j,1)+240),coord(j,2):(coord(j,2)+240),:,lim1(i):3:end) ,60);
       
        % save original resolution
        %load(sprintf('C:\\Users\\zxio506\\Desktop\\ECP\\Phi%d-%d.mat',i,j))
        [Xq,Yq,Zq] = meshgrid(linspace(1,60,60),linspace(1,60,60),linspace(1,size(Phi_actual,3),size(Phi_actual,3)*3));
        Phi = interp3(Phi_actual,Xq,Yq,Zq,'linear');
        [Xq,Yq,Zq] = meshgrid(linspace(1,60,240),linspace(1,60,240),linspace(1,size(Phi,3),size(Phi,3)));
        Phi = interp3(Phi,Xq,Yq,Zq,'spline');
        save(sprintf('C:\\Users\\zxio506\\Desktop\\ShowJichao\\Phi%d-%d',i,j),'Phi');
        
        % obtain trajectories
        Phi = Phi_actual;
        [Phase_Map] = ExtraCellularUnipolarPotential2PhaseMap(double(Phi));
        [PS_trajectory] = LocalizePhaseMapSingularity(Phase_Map);
        
        % save to file
        save(sprintf('C:\\Users\\zxio506\\Desktop\\ECP\\Phi%d-%d',i,j),'Phi');
        save(sprintf('C:\\Users\\zxio506\\Desktop\\ECP\\PS%d-%d',i,j),'PS_trajectory');
        
    end
end

%% Visualization
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