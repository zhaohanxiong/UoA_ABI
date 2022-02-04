function resampled_image = Resample_down(image) 

[~,~,Nz] = size(image);
for i= 1:3:Nz
    resampled_image(:,:,((i-1)/3)+1) = imresize(image(:,:,i),1/3,'nearest');
end
end