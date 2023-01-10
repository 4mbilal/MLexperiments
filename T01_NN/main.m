close all
clear all
clc

data_2d = zeros(1000,1000,3);
rng(1);% Set random generator to give same random numbers every time this code is run.
% idx_class0 = round((rand(125,2)-0.5)*250+500);
% idx_class0 = round((rand(125,2)-0.5)*250+[250,500]);
idx_class0 = round([(rand(125,2)-0.5)*250+[250,500];(rand(125,2)-0.5)*50+[500,100]]);
idx_class1 = [round(rand(75,2)*150+50);round(rand(75,2)*150+[800,50]);round(rand(75,2)*150+[50,800])];

data_2d = insertShape(data_2d,'circle',[idx_class0,5*ones(length(idx_class0),1)],'LineWidth',2,'Color', 'green');
data_2d = insertShape(data_2d,'circle',[idx_class1,5*ones(length(idx_class1),1)],'LineWidth',2,'Color', 'red');
imshow(data_2d)
% return

trainX = [idx_class0;idx_class1]';
trainX = trainX/1000; %feature scaling
trainY = [zeros(length(idx_class0),1);ones(length(idx_class1),1)]';

lr = 0.01;
iter = 50000;
minLoss = 1e-6;
nc = 3;
Mdl = fit2LayerNN(trainX',trainY',nc,lr,iter,minLoss);

% lr = 1e-2;
% iter = 600;
% minLoss = 1e-5;
% nc0 = 8;
% nc1 = 2;
% Mdl = fit3LayerNN(trainX',trainY',nc0,nc1,lr,iter,minLoss);

s = size(data_2d);
data_2d_indx = ones(1000,1000);
[testXc,testXr] = find(data_2d_indx);
testXc = testXc/1000;
testXr = testXr/1000;
testX = [ones(1,1e6);testXc';testXr'];

theta_1 = Mdl.l1;
theta_2 = Mdl.l2;
[labels,a1,z1] = fwd2NN(theta_1,theta_2,testX');
% theta_0 = Mdl.l0;
% [labels,a1,z1,a0,z0] = fwd3NN(theta_0,theta_1,theta_2,testX');

i=1;
for r = 1:1:s(1)
%     r
    for c = 1:1:s(2)
        label = labels(i);
        i = i+1;
        if(label<0.5)
            data_2d(r,c,:)=[0,0.5,0];%Green
        else
            data_2d(r,c,:)=[0.5,0,0];%Red
        end
    end
end
data_2d = insertShape(data_2d,'circle',[idx_class0,5*ones(length(idx_class0),1)],'LineWidth',2,'Color', 'green');
data_2d = insertShape(data_2d,'circle',[idx_class1,5*ones(length(idx_class1),1)],'LineWidth',2,'Color', 'red');

figure
imshow(data_2d)
