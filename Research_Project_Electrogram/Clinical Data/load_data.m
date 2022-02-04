%%
count = 0;
[n, m] = size(constellationsignal);
constellationbipolar = zeros(56,m);
for i = 1:8:64
    for j = 1:7
        count = count + 1;
        constellationbipolar(count,:) = constellationsignal(i+j,:)-constellationsignal(i+j-1,:);
    end
end

ECP = zeros([8,8,size(constellationsignal,2)]);

count = 0;
for i = 1:8
    for j = 1:8
        count = count + 1;
        ECP(i,j,:) = constellationsignal(count,:);
    end
end


[Xq,Yq,Zq] = meshgrid(linspace(1,7,60),linspace(1,8,60),linspace(1,size(ECP,3),size(ECP,3)));
ECP_interp = interp3(ECP,Xq,Yq,Zq,'linear');
ECP = ECP_interp;

save('C:\Users\zxio506\Desktop\clinical_data','ECP')

%%
close all;
t = 20000;
figure,imagesc(ECP(:,:,t));colormap('jet');
figure,imagesc(ECP_interp(:,:,t));colormap('jet');
figure,plot(squeeze(ECP(4,4,:)));

%%
close all;

for i = 1:size(ECP,3)
    imagesc(fliplr(ECP(:,:,i)));
    %caxis([0 2]);
    colormap('jet')
    pause(0.01)
end

%%
