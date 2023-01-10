function [y_pred,a1,z1] = fwd2NN(theta_1,theta_2,X)
    z1 = theta_1*X';
    a = tanh(z1);   %Tanh
    s = size(a);
    a1 = [ones(1,s(2));a];
    
    z2 = theta_2*a1;
    y_pred = 1./(1+exp(-z2));%Sigmoid
end
