mean=zeros(19200,1);
cov_matrix=zeros(19200,19200);

for i=1:16
    a=imread(sprintf("image_%d.png",i));
    a=cast(a,"double");
    a=reshape(a,[19200,1]);
    mean=mean+a;
    cov_matrix=cov_matrix+a*transpose(a);
end

mean=mean/16;
cov_matrix=cov_matrix/16;
cov_matrix=cov_matrix-mean*transpose(mean);
mean_plot=reshape(mean,80,80,3);
mean_plot=double(mean_plot)/double(255);

[eigenvectors,eigenvalues]=eigs(cov_matrix,10);
eigenvalues=eigs(cov_matrix,10);
image1=reshape(eigenvectors(:,1),80,80,3);
image2=reshape(eigenvectors(:,2),80,80,3);
image3=reshape(eigenvectors(:,3),80,80,3);
image4=reshape(eigenvectors(:,4),80,80,3);

image1=double(image1)/double(255);
image2=double(image2)/double(255);
image3=double(image3)/double(255);
image4=double(image4)/double(255);


figure;
subplot(1,5,1);
imshow(mean_plot);
subplot(1,5,2);
imshow(image1);
subplot(1,5,3);
imshow(image2);
subplot(1,5,4);
imshow(image3);
subplot(1,5,5);
imshow(image4);


for i=1:16
    a=imread(sprintf("image_%d.png",i));
    a=cast(a,"double");
    a=reshape(a,[19200,1]);
    closestimage=mean+dot(a,eigenvectors(:,1))*eigenvectors(:,1)+dot(a,eigenvectors(:,2))*eigenvectors(:,2)+dot(a,eigenvectors(:,3))*eigenvectors(:,3)+dot(a,eigenvectors(:,4))*eigenvectors(:,4);
    closestimage=reshape(closestimage,80,80,3);
    closestimage=double(closestimage)/double(255);
    a=reshape(a,80,80,3);
    a=double(a)/double(255);
    figure;
    imshow([a closestimage]);
end