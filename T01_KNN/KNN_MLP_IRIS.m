close all
clear all
clc

rng(0)

Data_orig = xlsread('Iris.csv');
%CSV files has three text labels (Setosa-1, Versicolor-2, Virginica-3)
%xlsread does not read text labels. So, manually add
Data_orig = [Data_orig(:,2:end) ones(150,1)];
for j=1:50,
    Data_orig(j,5) = 1;
end
for j=51:100,
    Data_orig(j,5) = 2;
end
for j=101:150,
    Data_orig(j,5) = 3;
end

% Data_orig = xlsread('ArtificialData_MultiClass.xlsx');

% Randomly shuffle the data
N = length(Data_orig);
idx = randperm(N) ;
X = Data_orig(idx',1:end-1);
Y = Data_orig(idx',end);

%Keep 80% examples for training
L = round(N*0.8);
Xtrain = X(1:L,:);
Xtest = X(L+1:end,:);
Ytrain = Y(1:L,:);
Ytest = Y(L+1:end,:);

% Train the KNN classifier
K = 3;      %K nearest neighbors
mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',K);
Ypred = predict(mdl,Xtest);
accuracy = sum(Ypred == Ytest)/numel(Ytest);
fprintf('KNN Accuracy: %.2f%%\n',accuracy*100);
% return

% Train the MLP classifier
Ytrain = ind2vec(Ytrain');%Convert labels to one-hot encoding
Ytest = ind2vec(Ytest');%Convert labels to one-hot encoding
net = patternnet(25);
view(net);
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0;
net.trainParam.min_grad = 1e-10;
net.trainParam.showWindow = true;
net = train(net,Xtrain',Ytrain);
Ypred = net(Xtest');

% Evaluate the performance of the classifier
[~,Ypred] = max(Ypred);
[~,Ytest] = max(Ytest);
accuracy = sum(Ypred == Ytest)/numel(Ytest);
fprintf('Accuracy: %.2f%%\n',accuracy*100);
% 


