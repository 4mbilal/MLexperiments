function [y_pred,a1,z1,a0,z0] = fwd3NN(theta_0,theta_1,theta_2,X)
    z0 = theta_0*X';
    a = tanh(z0);
    s = size(a);
    a0 = [ones(1,s(2));a];
    
    z1 = theta_1*a0;
    a = tanh(z1);
    s = size(a);
    a1 = [ones(1,s(2));a];
    
    z2 = theta_2*a1;
    y_pred = 1./(1+exp(-z2));   %=a2
end
