clear all
close all
clc

% X = xlsread('heart.csv');
% X = X(:,1:13);
% X = randn(303,13);
% return
X = [1.2 3.9 5.8 6.2;  3 9.75 14.5 15.5]'
% X = [2.5 0.5 2.2 1.9 3.1 2.3 2 1 1.5 1.1; 2.4 0.7 2.9 2.2 3 2.7 1.6 1.1 1.6 0.9]'
scatter(X(:,1),X(:,2))

mu = mean(X)
var = var(X)
X_scaled = X - mu
X_scaled = X_scaled./var.^0.5       %With scaling gives a better understanding of correlation
S = (X_scaled'*X_scaled)/(length(X)-1)
[eigvec,eigval] = eig(S)

Y = X_scaled*eigvec
% Y(:,1) = 0

figure
scatter(Y(:,2),(Y(:,1)))

Xreconstructed = ((Y*eigvec').*var.^0.5) + mu %with scaling
% Xreconstructed = ((Y*eigvec')) + mu %without scaling
mean(mean(100*abs(X- Xreconstructed)./X))

% [eigvec,~,eigval,~,~,mu] = pca(X)
