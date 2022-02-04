function resampled_image = Resample_new(image) 

[~,~,Nz] = size(image);
for i= 1:Nz
    resampled_image(:,:,3*i-2) = imresize(image(:,:,i),3,'nearest');
    resampled_image(:,:,3*i-1) = imresize(image(:,:,i),3,'nearest');
    resampled_image(:,:,3*i) = imresize(image(:,:,i),3,'nearest');
end
end