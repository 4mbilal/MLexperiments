clear all
close all
clc

rng(0);%pseudo-random generator seed

% Generate artificial training examples
x = [-10:0.1:10];   %one feature
y = -20 + 5.5*x;    %labels
y = y + randn(size(y))*10;
% plot(x,y)
scatter(x,y)
xlabel('X Values (Feature)')
ylabel('Y Values (Label)')

trainX = [ones(size(x));x];     %featuers of training examples
trainY = y;                     %labels of training examples
n = length(trainX);             %no. of training examples
theta = rand(1,2);
lr = 0.01;                     %learning rate
loss = [];

iter = 0;
%Solution using Gradient Descent Algorithm for Linear Regression
while(1)
    iter = iter + 1;
    h = theta*trainX;               %current hypothesis
    J = sum((h-trainY).^2)/(2*n);   %Cost function (MSE)
    dJ = (trainX*(h-trainY)')/n;    %partial gradients of Cost function using vectorized code
    theta = theta - lr*dJ';         %theta update 
    loss = [loss,J];                %loss/cost history for plotting
    
    if(rem(iter,100)==0) %Plot every 10 iterations only
        subplot(1,2,1)
        scatter(x,y)
        hold on
        plot(x,h)
        hold off
        ylabel('x (feature)')
        xlabel('y (label)')
        title('Linear regression line')
        subplot(1,2,2)
        plot(loss)
        ylabel('Loss / Cost')
        xlabel('iteration no.')
        title('Cost function vs. iterations')
        drawnow
    %     pause(0.5)
    end
    if(length(loss)>2)
    convg = abs(loss(end)-loss(end-1))/loss(end);
        if(convg<lr*1e-3)
            break;
        end
    end
    if(iter>1e4)
        break;
    end
end

theta_gd = theta

% %Solution using Normal Equation
n = length(x);
Sx = sum(x);
Sx2 = sum(x.^2);
Sy = sum(y);
Sxy = sum(x.*y);
X = [n Sx;Sx Sx2];
Y = [Sy;Sxy];
theta_normal_eq = inv(X)*Y

% Why Normal Equation method is not used in practice? Inverse is too slow.
% Also, it can only be used for polynomial hypothesis, not for deep NN.
% r = randn(4e3);
% tic
% invr = inv(r);
% toc

% Pseudo-inverse Method -19.7401 5.2872
theta_pseudo_inv = ((inv(trainX*trainX')*trainX))*trainY'



