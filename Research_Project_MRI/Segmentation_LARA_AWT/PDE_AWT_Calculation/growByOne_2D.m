% This function increases the size of a given mask or tif stack by one for each
% direction on each non zero point


function [GrownRegion] = growByOne_2D(Region)

if nargin > 1
    error('Too many input arguements')
else
    %get parameters
    [Nx, Ny, Nz] = size(Region);
    
    %preallocate output
    GrownRegion = uint8(zeros(Nx,Ny,Nz));
    %     GrownRegion = zeros(Nx,Ny,Nz);
    for i=2:Nx-1
        for j = 2:Ny-1
            for k = 2:Nz-1
                
                if Region(i,j,k) > 0
                    GrownRegion(i,j,k) = Region(i,j,k);
                    GrownRegion(i-1,j,k) = Region(i,j,k);
                    GrownRegion(i,j-1,k) = Region(i,j,k);
                    GrownRegion(i+1,j,k) = Region(i,j,k);
                    GrownRegion(i,j+1,k) = Region(i,j,k);
                end
            end
            
        end
    end
end

end