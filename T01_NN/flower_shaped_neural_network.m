clear all
close all
clc
rng(0)
% Generate 2D features
numPoints = 1000;
theta = linspace(0, 2*pi, numPoints);
r1 = 0.75 + 0.25*sin(5*theta); % Flower shape
r2 = 0.5 + 0.25*sin(5*theta); % Inverted flower shape

x1 = r1 .* cos(theta);
y1 = r1 .* sin(theta);
x2 = r2 .* cos(theta);
y2 = r2 .* sin(theta);

features = [x1', y1'; x2', y2'];
labels = ([ones(numPoints, 1); zeros(numPoints, 1)]);
idx = randperm(2*numPoints);
features = features(idx, :);
labels = categorical(labels(idx));

% Plot the features
figure;
scatter(x1, y1, 'r');
hold on;
scatter(x2, y2, 'b');
title('2D Features for Classification');
xlabel('X');
ylabel('Y');
legend('Class 1', 'Class 2');
hold off;

% net = MLP_sigmoid(features,labels); %Number of learnables: 52
net = MLP_relu(features,labels);%Number of learnables: 52
% net = MLP_relu_Three_Layers(features,labels);%Number of learnables: 317


% Generate a grid of points to visualize the decision boundary
[xGrid, yGrid] = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100));
gridPoints = [xGrid(:), yGrid(:)];
gridLabels = minibatchpredict(net,gridPoints);

[~, YPred] = max(gridLabels');

% Plot the decision boundary
figure;
gscatter(gridPoints(:,1), gridPoints(:,2), YPred-1, 'rb', 'xo');
hold on;
scatter(x1, y1, 'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1]);
scatter(x2, y2, 'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[1 0 0]);


title('Classification with Flower-Shaped Decision Boundary');
xlabel('X');
ylabel('Y');
legend('Class 1 Boundary', 'Class 2 Boundary', 'Class 1', 'Class 2');
hold off;


function net = MLP_sigmoid(features,labels)
layers = [
    featureInputLayer(2,"Name","featureinput")
    fullyConnectedLayer(10,"Name","fc")
    sigmoidLayer("Name","sigmoid")
    % reluLayer
    fullyConnectedLayer(2,"Name","fc_1")
    softmaxLayer("Name","softmax")];
net = dlnetwork(layers);

summary(net)%Number of learnables: 52
tic

options = trainingOptions("adam", ...
    MaxEpochs=50, ...
    MiniBatchSize=300, ...
    InitialLearnRate=3e-1, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=80, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

net = trainnet(features,labels,net,"crossentropy",options);

% MaxEpochs = 50;
% InitialLearnRate = 3e-1;
% MiniBatchSize = 250;
% 
% net = custom_training_loop(features,labels,net,MaxEpochs,InitialLearnRate,MiniBatchSize);
toc
end

function net = MLP_relu(features,labels)
layers = [
    featureInputLayer(2,"Name","featureinput")
    fullyConnectedLayer(10,"Name","fc")
    reluLayer()
    fullyConnectedLayer(2,"Name","fc_1")
    softmaxLayer("Name","softmax")];
net = dlnetwork(layers);

summary(net)%Number of learnables: 52
% options = trainingOptions("adam", ...
%     MaxEpochs=50, ...
%     MiniBatchSize=300, ...
%     InitialLearnRate=3e-1, ...
%     LearnRateSchedule="piecewise", ...
%     LearnRateDropFactor=0.1, ...
%     LearnRateDropPeriod=80, ...
%     Shuffle="every-epoch", ...
%     Plots="training-progress", ...
%     Metrics="accuracy", ...
%     Verbose=false);
% 
% net = trainnet(features,labels,net,"crossentropy",options);

MaxEpochs = 150;
InitialLearnRate = 2e-1;
MiniBatchSize = 250;

net = custom_training_loop(features,labels,net,MaxEpochs,InitialLearnRate,MiniBatchSize);

end

function net = MLP_relu_Three_Layers(features,labels)
layers = [
    featureInputLayer(2,"Name","featureinput")
    fullyConnectedLayer(15,"Name","fc")
    reluLayer()
    fullyConnectedLayer(15,"Name","fc1")
    reluLayer()
    fullyConnectedLayer(2,"Name","fc_2")
    softmaxLayer("Name","softmax")];
net = dlnetwork(layers);

summary(net)%Number of learnables: 52
% options = trainingOptions("adam", ...
%     MaxEpochs=150, ...
%     MiniBatchSize=400, ...
%     InitialLearnRate=2e-2, ...
%     LearnRateSchedule="piecewise", ...
%     LearnRateDropFactor=0.1, ...
%     LearnRateDropPeriod=80, ...
%     Shuffle="every-epoch", ...
%     Plots="training-progress", ...
%     Metrics="accuracy", ...
%     Verbose=false);
% 
% net = trainnet(features,labels,net,"crossentropy",options);

MaxEpochs = 150;
InitialLearnRate = 2e-2;
MiniBatchSize = 500;

net = custom_training_loop(features,labels,net,MaxEpochs,InitialLearnRate,MiniBatchSize);

end

function net = custom_training_loop(features,labels,net,MaxEpochs,InitialLearnRate,MiniBatchSize)
monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info=["Epoch" "LearnRate"], ...
    XLabel="Iterations");

averageGrad = [];
averageSqGrad = [];
learnRate = InitialLearnRate;
gradDecay = 0.9;
sqGradDecay = 0.999;

numIterationsPerEpoch = floor(numel(labels)/MiniBatchSize);
epoch = 0;
while epoch < MaxEpochs  && ~monitor.Stop
    epoch = epoch + 1;
    epoch_iter = 0;
    while epoch_iter < numIterationsPerEpoch  && ~monitor.Stop
        epoch_iter = epoch_iter + 1;

        [X,T] = miniBatchDataDispatcher(epoch_iter,MiniBatchSize,features,labels);

        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
        net.State = state;
        learnRate = learningRateSchedular(InitialLearnRate,epoch,MaxEpochs);
        global_iteration_count = (epoch-1)*numIterationsPerEpoch + epoch_iter;
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,global_iteration_count,learnRate,gradDecay,sqGradDecay);

        % Update the training progress monitor.
        recordMetrics(monitor,global_iteration_count,Loss=loss);
        updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
        monitor.Progress = 100 * global_iteration_count/(epoch*numIterationsPerEpoch);
    end
    acc = evaluate_accuracy(features,labels,net);
    fprintf('Epoch: %d\t Loss: %.2f\t Accuracy: %.2f\n', epoch,loss,acc*100);
end
end

function [loss,gradients,state] = modelLoss(net,X,T)
[Y,state] = forward(net,X);
loss = crossentropy(Y,T);
gradients = dlgradient(loss,net.Learnables);
end

function [X,T] = miniBatchDataDispatcher(epoch_iter,MiniBatchSize,features,labels)
    lower_idx = ((epoch_iter-1)*MiniBatchSize)+1;
    upper_idx = lower_idx + MiniBatchSize - 1;
    X = features(lower_idx:upper_idx,:);
    X = dlarray(X', 'CB');
    T = labels(lower_idx:upper_idx)';
    T = onehotencode(T,1);
    T = dlarray(single(T));
end

function learnRate = learningRateSchedular(initialLearnRate,epoch,MaxEpochs)
    % learnRate = initialLearnRate*(1-(epoch-1)/MaxEpochs);
    learnRate = initialLearnRate;
end

function acc = evaluate_accuracy(features,labels,net)
    X = dlarray(features', 'CB');
    [Y,state] = forward(net,X);
    [~, YPred] = max(Y);
    T = single(labels');
    acc = sum((YPred==(T)),"all")/numel(labels);
    % keyboard
end