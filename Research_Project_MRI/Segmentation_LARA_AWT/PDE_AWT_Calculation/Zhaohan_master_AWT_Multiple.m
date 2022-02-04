clear

Hearts = {'000010v2','000011'};

parent = pwd;

for H = 1:length(Hearts)
    
    Heart = Hearts{H};

    cd(parent)
    Zhaohan_Preprocessing
    clearvars -except Heart parent Hearts H
    cd(parent);

    Aclosing = './Atria_Closing_Mask/';
    createSurfaceMasks_New_6b
    clearvars -except Heart parent Hearts H
    cd(parent);

    createLayeredTissue_7
    clearvars -except Heart parent Hearts H
    cd(parent);

    LaplaceSolver_8
    clearvars -except Heart parent Hearts H
    cd(parent);

    MiddleLine_9
    clearvars -except Heart parent Hearts H
    cd(parent);

    LaplaceSolverwithMidline_10
    clearvars -except Heart parent Hearts H
    cd(parent);

    createGradientField_11
    clearvars -except Heart parent Hearts H
    cd(parent);

    section = 0;
    AWT_PDE_MarkoVersion_12
    if iter == 200
        failed0 = 1;
        save failed0.mat failed0
    end
    clearvars -except Heart parent Hearts H
    cd(parent);

    section = 1;
    AWT_PDE_MarkoVersion_12
    if iter == 200
        failed1 = 1;
        save failed1.mat failed1
    end
    clearvars -except Heart parent Hearts H
    cd(parent);

    section = -1;
    maskPath = './Atria_Closing_Mask/';
    AWT_PDE_MarkoVersion_12
    clearvars -except Heart parent Hearts H
    cd(parent);

    Resolution = 0.625;
    WallThicknessResults_13
    clearvars -except Heart parent Hearts H

    load(strcat('Unrounded_AWT_',Heart,'_results__PDE.mat'))
    load EndoToEndoLaplaceFieldLines.mat
    load indexs.mat

    %AWTfinal(tissue == 200) = 0; 

    temp = zeros(Nxo,Nyo,Nzo);
    temp(index_array(1):index_array(2),index_array(3):index_array(4),index_array(5):index_array(6)) = AWTfinal;
    cd(parent);
    temp = Resample_down(temp);
    Nz = 176;
    thickness = temp(:,:,10:Nz+9);

    cd(Heart);
    AWT = thickness;
    save('AWT.mat','AWT');

end

