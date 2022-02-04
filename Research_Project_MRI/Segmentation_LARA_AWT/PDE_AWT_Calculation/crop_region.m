function [shrunk_region,indexarray] = crop_region(region)

[Nx,Ny,Nz] = size(region);
startX = -1;
finishX= -1;
startY= -1;
finishY= -1;
startZ= -1;
finishZ= -1;
for i = 1:Nz
    if max(max(region(:,:,i))) > 0 && startZ == -1
        startZ = i;
    elseif max(max(region(:,:,i))) == 0 && startZ ~= -1
        finishZ  = i-1;
        break
    end
end

for i = 1:Ny
    if max(max(region(:,i,:))) > 0 && startY == -1
        startY = i;
    elseif max(max(region(:,i,:))) == 0 && startY ~= -1
        finishY  = i-1;
        break
    end
end

for i = 1:Nx
    if max(max(region(i,:,:))) > 0 && startX == -1
        startX = i;
    elseif max(max(region(i,:,:))) == 0 && startX ~= -1
        finishX  = i-1;
        break
    end
end

padding = 9;

shrunk_region = region(startX-padding:finishX+padding,startY-padding:finishY+padding,startZ-padding:finishZ+padding);
indexarray = [startX-padding,finishX+padding,startY-padding,finishY+padding,startZ-padding,finishZ+padding];
end

