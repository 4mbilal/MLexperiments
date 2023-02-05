clear all
close all
clc

[X] = read_orl_faces('')';%400 images for 40 people
[eigvec,~,eigval,~,~,mu] = pca(X,'Economy',false);

z  = [ reshape(eigvec(:,1),112,92)  reshape(eigvec(:,2),112,92)   reshape(eigvec(:,3),112,92) ; reshape(eigvec(:,4),112,92)   reshape(eigvec(:,5),112,92)   reshape(eigvec(:,6),112,92)];
figure
imshow(z,[],'Initialmagnification','fit');
title('Six most significant Eigen Faces')
figure
imshow(reshape(mu,112,92),[],'Initialmagnification','fit');
title('Mean Face :)')

s = size(X);

nsel=400;      % Number of eigen faces (feature dimensions) to keep
Xform = eigvec(:,1:nsel);
feature_vectors_compressed = (X-mu)*Xform;
feature_vectors_reconstructed = uint8((feature_vectors_compressed*Xform')+mu);
% pause

figure
  for i=1:s(1)
      z = [uint8(reshape(feature_vectors_reconstructed(i,:),112,92)) reshape(X(i,:),112,92)];
      imshow(z,[],'Initialmagnification','fit')
%       title(num2str(Y(i)))
      drawnow
      pause(1)
  end

