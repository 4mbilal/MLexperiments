clear all
close all
clc

% data  = xlsread('LinearRegressionTrainingData_3f.xlsx');
data  = xlsread('train_subset.csv');
data(isnan(data)) = 0;
nf = 36; %Number of features

%Training Linear Regression Model
thetasPred = randn(nf+1,1);
lr = 0.2;  %critical (start from low value)
loss = [];
trainX = data(:,1:nf); 
maxV = max(abs(trainX));
trainX = trainX./maxV;
s = size(trainX);
trainX = [ones(s(1),1),trainX];
trainY = data(:,nf+1);

cnt = 1;
while(cnt<100000)
    Y = trainX*thetasPred;
    Grad = (trainX'*(Y-trainY))/length(trainX);
    thetasPred = thetasPred-lr*Grad;
    loss = [loss sum((Y-trainY).^2)/length(trainX)];
    if(rem(cnt,1000)==0)
    plot(loss)
    drawnow
    end
%     pause(0.01)
    cnt = cnt + 1;
% pause
end


data  = xlsread('test_subset.csv');
% data  = xlsread('LinearRegressionTestData_3f.xlsx');
data(isnan(data)) = 0;
testX = data(:,1:nf); 
testX = testX./maxV;
testY = data(:,nf+1);

% testX = [-114.5961177	11.32021162	22.14561896];
s = size(testX);
testX = [ones(s(1),1),testX];
YPred = testX*thetasPred;

mse = mean((YPred-testY).^2)


