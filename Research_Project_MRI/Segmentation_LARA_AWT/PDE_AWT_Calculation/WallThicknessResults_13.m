%13
%% Wall Thickness Results
tic
parent = pwd;
cd(parent)
cd([Heart, '/']);
%Use resolution and load the remaining data.
removalPercentage = 0.001; % can adjust this parameter
load parameters.mat % From Noise Removal (1)
approach = 1;
disp ('loading data: make sure correct thickness data is in code');
if approach == 1
%load PDE data
    thicknessData = 'Thickness.mat' % change accordingly
elseif approach == 0
    thicknessData = 'AWT_H6425_21-Oct-2020_PDE.mat' % change accordingly
else
    disp('incorrect number entered');
    return
end
load(thicknessData);

% load BinarySegmentedAtria.mat %From MaskProcessing (2)
% load AWT_H66_combined_22Jan.mat % From CombineWallThickness (13) + Applying Unmasking (H66_unmasking22jan)

% load AWT_H66_combined_25Jan_unmasked.mat % From CombineWallThickness (13) + Applying Unmasking (MaskRemovalwithAppendages.mat)
load EndoToEndoLaplaceFieldLines.mat % From LaplaceSolver (8)
load 3DCavityLAendo.mat % From regionGrowingProcessing (5)
load 3DCavityRAendo.mat % From regionGrowingProcessing (5)

%Get the dimensions needed for the loops by checking the first file
%Nz = length(files);
maskAppendage= uint8(zeros(Nx,Ny,Nz));
%[Ny,Nx] = size(maskAppendage(:,:,1));
%Read the rest of the images
disp('saving appendage mask');
% save appendageMask.mat maskAppendage

query = 0;

if query == 1
    disp('Reading extra mask: MAKE SURE FOLDER NAME IS CORRECT');
    path = './H121_V2_ExtraMask/' % change accordingly
    files = dir([path '*.tif']);
    %Get the dimensions needed for the loops by checking the first file
    %Nz = length(files);
    valveMask= uint8(zeros(Nx,Ny,Nz));
    %[Ny,Nx] = size(maskAppendage(:,:,1));
    %Read the rest of the images
    for i=1:Nz
        valveMask(:,:,i) = imread([path files(i).name]);
    end 
end
load atriaClosingMask.mat %From processClosedAtria (4)
%% For surfaces only approach
% load epi_surface.mat
% load endoLA_surface.mat
% load endoRA_surface.mat

%%
load newMiddleLine.mat %From MiddleLine (9)
%remove interpolated data points that have decimals
disp('Unmasking');
cd(parent);
if approach == 1
    AWTfinal = Thickness_unmasked;
end
AWTfinal(Middle > 0) = 0;
% AWTfinal = round(AWTfinal);
AWTfinal = AWTfinal*Resolution;

%remove values from appendage

AWTfinal(maskAppendage>0)=0;

if query == 1
    AWTfinal(valveMask > 0) = 0;
    clear valveMask
end
%disp('Reading hole fill mask: MAKE SURE FOLDER NAME IS CORRECT');

%Remove mask values ALREADY DONE IN 12
% mask2 = growByOne(mask);
% mask3 = growByOne(mask2);
% mask4 = growByOne(mask3);
% clear mask mask2 mask3
% AWTfinal(mask4 == 2)=0;

%AWTfinal(dataCleanedBinary == 0) = 0;
%% Remove upper and lower Limits dependng on the percentages
if removalPercentage ~= 0
    disp(['Removing the ', num2str(removalPercentage), ' percentile values from the results']);
    wholeAtriaordered = sort(AWTfinal(AWTfinal~=0));
    index = round(length(wholeAtriaordered)*0.01*removalPercentage);
    endIndex = length(wholeAtriaordered)-index;
    lowerLimit = wholeAtriaordered(index);
    upperLimit = wholeAtriaordered(endIndex);
    AWTfinal(AWTfinal>=upperLimit) = 0;
    AWTfinal(AWTfinal<=lowerLimit) = 0;
else
    wholeAtriaordered = sort(AWTfinal(AWTfinal~=0));
    lowerLimit = 0;
    upperLimit = inf;
end
%% do RA - rename to RA if doing whole tissue approach
disp('Calculating Right Atria Results');
RA = tissue;
RA(tissue <= 200) = 0;
%% For surfaces only approach
% RA1(endoRACavity > 0)=0;
% RA2 = RA1;
% RA2(endoRA_surface ==0)=0;
% newEpi = growByOne(epi_surface);
% newEpi2 = growByOne(newEpi);
% newEpi = growByOne(newEpi2);
% newEpi2 = growByOne(newEpi);
% RA1(newEpi2 == 0)=0;
% RA = RA1 + RA2;

%% Calculate values
AWTRA = AWTfinal;
AWTRA(RA==0)=0;


RAordered = sort(AWTRA(AWTRA~=0));

RAtissueVolume = length(RAordered)*Resolution^3
RAcavityVolume = nnz(endoRACavity)*Resolution^3
% figure, histogram(RAordered);



raMean=mean(RAordered)
raMax=max(RAordered)
raMin=min(RAordered)
raMed=median(RAordered)
raSTD=std(RAordered)
raUQ=quantile(RAordered,0.75)
raLQ=quantile(RAordered,0.25)
raSE = std(RAordered)/sqrt(length(RAordered))

%% do LA - rename to LA1 if using whole tissue approach
disp('Calculating Left Atria Results');

LA=tissue;
LA(tissue>=200)=0;
%% For surfaces only approach
% LA1(endoLACavity > 0)=0;
% LA2 = LA1;
% LA2(endoLA_surface ==0)=0;
% newEpi = growByOne(epi_surface);
% newEpi2 = growByOne(newEpi);
% newEpi = growByOne(newEpi2);
% newEpi2 = growByOne(newEpi);
% LA1(newEpi2 == 0)=0;
% LA = LA1 + LA2;
%% Calculate values
AWTLA=AWTfinal;
AWTLA(LA==0)=0;


LAordered = sort(AWTLA(AWTLA~=0));
LAcavityVolume = nnz(endoLACavity)*Resolution^3
LAtissueVolume = length(LAordered)*Resolution^3

% 
% figure, histogram(nonzeros(LAordered));

laMean=mean(LAordered)
laMax=max(LAordered)
laMin=min(LAordered)
laMed=median(LAordered)
laSTD=std(LAordered)
laSE = std(LAordered)/sqrt(length(LAordered))
laUQ=quantile(LAordered,0.75)
laLQ=quantile(LAordered,0.25)

%% do Whole atria
disp('Calculating Whole Atria Results');

wholeAtriaordered = AWTfinal;
wholeAtriaordered(tissue == 200) = 0;
wholeAtriaordered = sort(nonzeros(wholeAtriaordered));

WholeAtriatissueVolume = length(wholeAtriaordered)*Resolution^3
WholeAtriacavityVolume = LAcavityVolume + RAcavityVolume

% figure, histogram(wholeAtriaordered);

wholeAtriaMean=mean(wholeAtriaordered)
wholeAtriaMax=max(wholeAtriaordered)
wholeAtriaMin=min(wholeAtriaordered)
wholeAtriaMed=median(wholeAtriaordered)
wholeAtriaSTD=std(wholeAtriaordered)
wholeAtriaUQ=quantile(wholeAtriaordered,0.75)
wholeAtriaLQ=quantile(wholeAtriaordered,0.25)
wholeAtriaSE = std(wholeAtriaordered)/sqrt(length(wholeAtriaordered))


%% Plot mean thickness per slice

for i = 1:Nz
    WAmeanPlot(i) = mean(mean(nonzeros(AWTfinal(:,:,i))));
    LAmeanPlot(i) = mean(mean(nonzeros(AWTLA(:,:,i))));
    RAmeanPlot(i) = mean(mean(nonzeros(AWTRA(:,:,i))));
end

% figure, plot(WAmeanPlot);
% hold on
% plot(LAmeanPlot);
% plot(RAmeanPlot);
% legend('WA', 'LA', 'RA');
% title('Mean Wall Thickness per Slice');
% xlabel('Slice Number');
% ylabel('Thickness (mm)')
% hold off


%% Save data
cd([Heart, '/']);
disp('Saving results');
if approach == 1
    save(['Unrounded_AWT_', Heart, '_results_','_PDE.mat'], 'AWTfinal', 'LAordered', 'RAordered', 'wholeAtriaordered', 'removalPercentage', 'lowerLimit' , 'upperLimit','-v7.3');
else
    save(['Unrounded_AWT_', Heart, '_results_','_old.mat'], 'AWTfinal', 'LAordered', 'RAordered', 'wholeAtriaordered', 'removalPercentage', 'lowerLimit' , 'upperLimit');    
end
toc
disp('AWT Results Complete');
disp('Run Colormap and Surface Generation Script (14)');