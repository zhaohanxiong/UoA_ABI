function [ region, surface ] = MyRegionGrowFunc( tissue, seed, stopAtBoundary, stop_step )
% finds region and surface bound from seeded point(s)

% check stopAtBoundary is valid
if ~((stopAtBoundary == 1)||(stopAtBoundary == 0))
    disp('stopAtBoundary must be 1 (true) or 0 (false)');
    region = 0;
    surface = 0;
    return
end

[Ny, Nx, Nz] = size(tissue);

surface = zeros(Ny, Nx, Nz);
region = zeros(Ny, Nx, Nz);

% initial 'region' with seed
if length(seed(:,1))>1
    for i=1:1:length(seed(:,1))
        region(seed(i,1),seed(i,2),seed(i,3)) = 1;
    end
else
    region(seed(1),seed(2),seed(3)) = 1;
end

neighbour = [1 0 0;-1 0 0;0 1 0;0 -1 0;0 0 1;0 0 -1];

change = 1;
count = 0;
count2 = 0;

step = 0;

% continue until there is no more change
while change~=0
    change = 0;
    
    % get points to check   
    temp = (surface==0) + (surface==1);
    [indY, indX, indZ] = ind2sub(size(region), find(region.*temp));
    
    if (length(indY)==0)
        disp('Done: no more points to check');
        surface(surface==2)=1;
        surface(surface>0) = 1;
        return
    end
    
    disp(sprintf('Y:%0.f,%0.f / X:%0.f,%0.f / Z:%0.f,%0.f / step:%0.f',min(indY),max(indY),min(indX),max(indX),min(indZ),max(indZ),step));
    
    % check points
    for i=1:1:length(indY)
        count = 0;
        count2 = 0;
        
        % check neighbour points
        for j=1:1:6
            iy = indY(i) + neighbour(j,1);
            ix = indX(i) + neighbour(j,2);
            iz = indZ(i) + neighbour(j,3);
            
            % check if within image
            if (iy>0) && (ix>0) && (iz>0) && (iy<=Ny) && (ix<=Nx) && (iz<=Nz)
                % check if tissue present
                if (tissue(iy,ix,iz)==0)
                    % check if point is new
                    if (surface(iy,ix,iz)==0)
                        % then set to 1
                        region(iy,ix,iz) = 1;
                        surface(iy,ix,iz) = 1;
                        change = 1;
                    else
                        count = count + 1;
                    end
                else
                    surface(iy,ix,iz) = 3;
                    count2 = count2 + 1;
                    change = 3;
                end
            else
                count = count + 1;
                % has hit boundary
                if stopAtBoundary
                    disp('Hit boundary.');
                    disp(sprintf('Last point tested:(%0.f,%0.f,%0.f)',indY(i),indX(i),indZ(i)));
                    surface(surface==2)=1;
                    surface(surface>0) = 1;
                    return
                end
            end
            
        end 
        
        %if all neighbours to point has been checked, do not need to check
        %this point again.
        if count==6
            % this means this point is inside
            surface(indY(i),indX(i),indZ(i)) = -1;
            change = 1;
        end
        
        if (count + count2)==6
            % this means this point is a surface point 
            surface(indY(i),indX(i),indZ(i)) = 2;
        end
        
    end
    
    step = step + 1;
    
    if step==stop_step
        disp('stop step reached');
        surface(surface==2)=1;
        surface(surface>0) = 1;
        return
    end
    
end

disp('Done: no more change detected');
surface(surface==2)=1;
surface(surface>0) = 1;

end
