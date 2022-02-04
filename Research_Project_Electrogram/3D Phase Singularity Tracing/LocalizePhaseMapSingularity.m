function [Trajectory] = LocalizePhaseMapSingularity(PhaseMap)

% input, the 2D phase map over time (3D)
% output, the trajectory

%% Nested Function
    function [result] = phase_change(a,b)
        if abs(a-b) <= pi
            result = a - b;
        elseif a - b > 0
            result = a - b - (2*pi);
        else
            result = a - b + (2*pi);
        end
    end

%% calculate the trajectory
Trajectory = cell(size(PhaseMap,3),1);
PhaseChange = zeros(size(PhaseMap,1),size(PhaseMap,2));

% loop through each time step, search each 3x3 region
for t = 1:size(PhaseMap,3)

    for x = 2:size(PhaseMap,1)-1
        for y = 2:size(PhaseMap,2)-1
            if sum(isnan(PhaseMap((x-1):(x+1),(y-1):(y+1),t))) == 0
                % obtain indices to evaluate line integral
                ii = [x-1,x-1,x-1,x,x+1,x+1,x+1,x];
                jj = [y-1,y,y+1,y+1,y+1,y,y-1,y-1];
                ii = [ii, ii(1)];
                jj = [jj, jj(1)];

                % calculate total phase change at this point
                result = 0;
                for ind = 1:8
                    result = result + phase_change( PhaseMap(ii(ind),jj(ind),t),PhaseMap(ii(ind+1),jj(ind+1),t) );
                end
                PhaseChange(x,y) = result;
            end
        end
    end

    % the positions where phase change is +- 2pi means a singularity exists
	PhaseChange(abs(abs(PhaseChange)-(2*pi)) > 0.001) = 0;
    PhaseChange(PhaseChange~=0) = 1;
    PhaseChange(isnan(PhaseMap(:,:,t))) = 0;
    cc = bwconncomp(PhaseChange);
    S = regionprops(cc,'Centroid');
    
    if isempty(S)
        Trajectory{t} = [nan,nan];
    else
        temp = [];
        for i = 1:length(S)
           temp = [temp ; S(i).Centroid];
        end
        Trajectory{t} = temp;
    end
end

end