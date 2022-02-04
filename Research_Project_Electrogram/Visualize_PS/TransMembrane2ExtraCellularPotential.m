function [Phi] = TransMembrane2ExtraCellularPotential(Vm,n_elect)

% Input:
%       Vm = the input data from VTK2MatlabSimpleCase generated from C
%       code. This input is the transmembrane potential.
%       n_elect = n x n virtual electrodes
%
% Output:
%       Phi = data converted to extracellular potential.
% 
% To Run: load Vm.mat;[Phi] = TransMembrane2ExtraCellularPotential(Vm);
%
% vectorized=0 runtime: 450s
% vectorized=1 runtime: 73s
% vectorized=1.1 runtime: 75s                               
% vectorized=2 runtime: 

%% paramters
vectorize = 1;  % which implementation mode to use

d = 0.5;        % the distance (in mm) from source to field point
spacing = 0.5;  % distance (in mm) of the discretization space
K = 0.33;       % ratio of intra & extra cellular conductivity

%% set up source of simulated atrial activity
Vm = single(Vm);
[x,y,z,tt] = size(Vm); % dimensions of simulated transmembran potential

x=single(x);y=single(y);z=single(z);tt=single(tt); % convert everything to single

xx = (0:(x-1))*spacing; % position in x direction
yy = (0:(y-1))*spacing; % position in y direction
zz = (0:(z-1))*spacing; % position in z direction

%% set up vitual eletrodes
step_x = round(x/n_elect); % calculate step size to sample n_electrodes in x
step_y = round(y/n_elect); % calculate step size to sample n_electrodes in y

xx_dash = xx(round(step_x/2):step_x:end);   % position in x direction
yy_dash = yy(round(step_y/2):step_x:end);   % position in y direction
zz_dash = max(zz)+d;                        % position in z direction (above source)

Phi = single(zeros(length(xx_dash),length(yy_dash),tt)); %  initialize extra cellular potential

%% converting from transmembrane to extra cellular
[xx,yy,zz] = meshgrid(xx,yy,zz); % vectorize r for each electrode

if vectorize == 1 % vectorized over time for each electrode
    
    [Vm_fx,Vm_fy] = gradient(Vm);

    for x = 1:length(xx_dash)
        for y = 1:length(yy_dash)
            
            fprintf('%d,%d\n',x,y);
            
            r_inv = ( (xx_dash(x)-xx).^2+(yy_dash(y)-yy).^2+(zz_dash-zz).^2 ).^-0.5;
            [r_fx,r_fy] = gradient(r_inv);

            Phi(x,y,:) = sum(sum(sum( bsxfun(@times,Vm_fx,r_fx) + bsxfun(@times,Vm_fy,r_fy) )));

        end
    end
    
elseif vectorize == 1.1 % vectorized over time for each electrode, vectorize distance calcs
    
    [Vm_fx,Vm_fy] = gradient(Vm);
        
    [xx_mesh,yy_mesh,zz_mesh] = meshgrid(xx_dash,yy_dash,zz_dash);
    r_x = single(reshape( pdist2(xx_mesh(:),xx(:)) ,n_elect,n_elect,size(xx,1),size(xx,2),size(xx,3)));
    r_y = single(reshape( pdist2(yy_mesh(:),yy(:)) ,n_elect,n_elect,size(yy,1),size(yy,2),size(yy,3)));
    r_z = single(reshape( pdist2(zz_mesh(:),zz(:)) ,n_elect,n_elect,size(zz,1),size(zz,2),size(zz,3)));
    r = permute( (r_x.^2+r_y.^2+r_z.^2).^-0.5 ,[3,4,5,1,2]);
    [r_fx,r_fy] = gradient(r);
    
    for x = 1:length(xx_dash)
        for y = 1:length(yy_dash)
            Phi(x,y,:) = sum(sum(sum( bsxfun(@times,Vm_fx,r_fx(:,:,:,x,y))+bsxfun(@times,Vm_fy,r_fy(:,:,:,x,y)) )));
        end
    end
    
elseif vectorize == 2 % fully vectorized (memory over flow on normal computer)
    
    [Vm_fx,Vm_fy] = gradient(permute(Vm,[4,1,2,3]));
    Vm_fx = repmat(Vm_fx,1,1,1,1,n_elect,n_elect);
    Vm_fy = repmat(Vm_fy,1,1,1,1,n_elect,n_elect);
    
    [xx_dash,yy_dash,zz_dash] = meshgrid(xx_dash,yy_dash,zz_dash);
    r_x = single(reshape( pdist2(xx_dash(:),xx(:)) ,n_elect,n_elect,size(xx,1),size(xx,2),size(xx,3)));
    r_y = single(reshape( pdist2(yy_dash(:),yy(:)) ,n_elect,n_elect,size(yy,1),size(yy,2),size(yy,3)));
    r_z = single(reshape( pdist2(zz_dash(:),zz(:)) ,n_elect,n_elect,size(zz,1),size(zz,2),size(zz,3)));
    r = reshape( permute( sqrt(r_x.^2+r_y.^2+r_z.^2) ,[3,4,5,1,2]) ,1,size(r,1),size(r,2),size(r,3),size(r,4),size(r,5));
    [r_fx,r_fy] = gradient(1./r);
    
    Phi = squeeze(sum(sum(sum( bsxfun(@times,Vm_fx,r_fx)+bsxfun(@times,Vm_fy,r_fy) ,2),3),4));
    Phi = permute(Phi,[2,3,1]);
    
else % not vectorized

    [Vm_fx,Vm_fy] = gradient(Vm);                                                   %grad(Vm)
    for x = 1:length(xx_dash)                                                       %  loop through virtual electrodes in x
        for y = 1:length(yy_dash)                                                   %  loop through virtual electrodes in y
            
            for t = 1:tt                                                            %  loop through all time steps    
                r = sqrt( (xx_dash(x)-xx).^2+(yy_dash(y)-yy).^2+(zz_dash-zz).^2 );  %r
                [r_fx,r_fy] = gradient(1./r);                                       %grad(1/r)
                Phi(x,y,t) = sum(sum(sum(Vm_fx(:,:,:,t).*r_fx+Vm_fy(:,:,:,t).*r_fy)));%dot prod & integr(dv)
            end
            
        end
    end
    
end

Phi = -K.*Phi;

return