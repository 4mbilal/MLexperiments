close all
clear all
clc

% Adapted from Matlab example at the following link
% https://www.mathworks.com/help/stats/fitcnet.html

% rng(10);
% Load the data
data = readtable('Iris.csv');
% Ignore the first column
data(:,1) = [];
% Load the first four columns as features
features = table2array(data(:,1:4));
% Load the last column as class labels
labels = data(:,5);

% Convert the labels to categorical if they are not
if ~iscategorical(labels)
    labels = categorical(table2array(labels));
end

% Get the number of examples
num_examples = size(features, 1);
% Generate a random permutation of indices
indices = randperm(num_examples);
% Use these indices to shuffle the features and labels
features = features(indices,:);
labels = labels(indices,:);

% labels_val_numeric = grp2idx(labels);

% Create a cvpartition object that defines the random 80-20 split
cvp = cvpartition(labels, 'HoldOut', 0.2);
% Extract the training features and labels
trainX = features(cvp.training,:);
trainY = labels(cvp.training,:);
% Extract the validation features and labels
testX = features(cvp.test,:);
testY = labels(cvp.test,:);

%feature scaling (works even without scaling because the dataset is simple)
% scaleX = repmat(max(abs(trainX)),length(trainY),1);
% trainX = trainX./scaleX; 
% scaleX = scaleX(1:length(testY),:);
% testX = testX./scaleX; 

% Try different number of neurons and number of layers to get a reasonable model
net = fitcnet(trainX,trainY,"LayerSizes",[8 8],"Verbose",1);
predictedlabelsTest = net.predict(testX);
AccuracyTest = 100*sum(double(testY==predictedlabelsTest))/length(testY)
predictedlabelsTrain = net.predict(trainX);
AccuracyTrain = 100*sum(double(trainY==predictedlabelsTrain))/length(trainY)

confusionchart(predictedlabelsTest,testY)
iteration = net.TrainingHistory.Iteration;
trainLosses = net.TrainingHistory.TrainingLoss;

figure
plot(iteration,trainLosses)
xlabel("Iteration")
ylabel("Cross-Entropy Loss")

