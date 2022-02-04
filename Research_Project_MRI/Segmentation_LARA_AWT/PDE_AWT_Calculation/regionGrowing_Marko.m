% Region growing; adapted from Dr Kroon? online
% Improve algorithm by only adding to queue if its in region

function J = regionGrowing_Marko(I, sX, sY, sZ, Resolution)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% I is binary imagez
% sX-Z is seed point
% Resolution is in cm
sizeI = size(I);
J = false((sizeI));

queue = -1 * ones(10000, 3);
i = 1;
n = 2;
queue(1,:) = [sX sY sZ];

count = 1;
printVol = 1;

while ( i < n )
    x = queue(i,1);
    y = queue(i,2);
    z = queue(i,3);
    

    if ( ~I(x,y,z) && ~J(x,y,z) )
        J(x,y,z) = 1;
        count = count + 1;
        
        if ( x > 1 )
            queue(n,1) = x - 1;
            queue(n,2) = y;
            queue(n,3) = z;
            n = n + 1;
        end
        if ( y > 1 )
            queue(n,1) = x;
            queue(n,2) = y - 1;
            queue(n,3) = z;
            n = n + 1; 
        end
        if ( z > 1 )
            queue(n,1) = x;
            queue(n,2) = y;
            queue(n,3) = z - 1;
            n = n + 1; 
        end
        if ( x < sizeI(1) )
            queue(n,1) = x + 1;
            queue(n,2) = y;
            queue(n,3) = z;
            n = n + 1; 
        end
        if ( y < sizeI(2) )
            queue(n,1) = x;
            queue(n,2) = y + 1;
            queue(n,3) = z;
            n = n + 1;
        end
        if ( z < sizeI(3) )
            queue(n,1) = x;
            queue(n,2) = y;
            queue(n,3) = z + 1;
            n = n + 1;
        end
        
%         if (count * Resolution^3 > printVol)
%             fprintf('Volume of region: %.2f cm^3.\n', count * Resolution^3);
%             printVol = printVol + 1;
%             imagesc(I(:,:,75) + 2 * J(:,:,75));
%             drawnow;
%         end
    end
    
    i = i + 1;
    
    if (length(queue) - n < 10)
        queue = [queue; -1 * ones(10000, 3)];
    end
    
    if (mod(i, 10000) > 9990)
        queue = queue(9001:end,:);
        i = i - 9000;
        n = n - 9000;
    end
    
    if count * Resolution^3 > 100
        disp('Region growing failed; leaks present. saving output for reference');
        break
    end
end

fprintf('Volume of region: %.2f cm^3.\n', count * Resolution^3);

end

