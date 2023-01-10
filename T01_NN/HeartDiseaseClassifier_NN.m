close all
clear all
clc

rng(10);
trainData = xlsread('heart_train.csv');
trainY = trainData(:,end)';
trainX = trainData(:,1:end-1);
testData = xlsread('heart_test.csv');
testY = testData(:,end)';
testX = testData(:,1:end-1);
%feature scaling
scaleX = repmat(max(abs(trainX)),length(trainY),1);
trainX = trainX./scaleX; 
scaleX = scaleX(1:length(testY),:);
testX = testX./scaleX; 

MatlabNN(trainX,trainY,testX,testY)
% custom2LNN(trainX,trainY,testX,testY)

%Overfitting Example!
% custom3LNN(trainX,trainY,testX,testY)

function MatlabNN(trainX,trainY,testX,testY)
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% hiddenLayerSize = [10 10 10 10];
hiddenLayerSize = [24];
net = fitnet(hiddenLayerSize,trainFcn);
view(net)
pause
[net,tr] = train(net,trainX',trainY,'UseGPU','yes');
labels = double(net(testX')>0.5);
AccuracyTest = 100*sum(double(testY==labels))/length(testY)
labels = double(net(trainX')>0.5);
AccuracyTrain = 100*sum(double(trainY==labels))/length(trainY)
end

function custom2LNN(trainX,trainY,testX,testY)
lr = 7e-2;
iter = 6000;
minLoss = 1e-7;
nc = 24;
Mdl = fit2LayerNN(trainX,trainY',nc,lr,iter,minLoss);

theta_1 = Mdl.l1;
theta_2 = Mdl.l2;
testX = [ones(size(testY')),testX];
[labels,a1,z1] = fwd2NN(theta_1,theta_2,testX);
labels = double(labels>0.5);
AccuracyTest = 100*sum(double(testY==labels))/length(testY)

trainX = [ones(size(trainY')),trainX];
[labels,a1,z1] = fwd2NN(theta_1,theta_2,trainX);
labels = double(labels>0.5);
AccuracyTrain = 100*sum(double(trainY==labels))/length(trainY)
end

function custom3LNN(trainX,trainY,testX,testY)
lr = 1e-4;
iter = 420;
minLoss = 1e-5;
nc0 = 64;
nc1 = 128;
lambda = 0; %Try 0.35
Mdl = fit3LayerNN(trainX,trainY',testX,testY',nc0,nc1,lr,iter,minLoss,lambda);
% Mdl = [];
% load('Mdl.mat','Mdl');
theta_1 = Mdl.l1;
theta_2 = Mdl.l2;
% [labels,a1,z1] = fwd2NN(theta_1,theta_2,testX');
theta_0 = Mdl.l0;
testX = [ones(size(testY')),testX];
[labels,a1,z1,a0,z0] = fwd3NN(theta_0,theta_1,theta_2,testX);
labels = double(labels>0.5);
AccuracyTest = 100*sum(double(testY==labels))/length(testY)

trainX = [ones(size(trainY')),trainX];
[labels,a1,z1,a0,z0] = fwd3NN(theta_0,theta_1,theta_2,trainX);
labels = double(labels>0.5);
AccuracyTrain = 100*sum(double(trainY==labels))/length(trainY)
end

