function Mdl = fit2LayerNN(X,Y,nc,lr,iter,minLoss)
    X = [ones(size(Y)),X];  
    s = size(X);
    theta_1 = randn(nc,s(2))*1;
    theta_2 = randn(1,nc+1)*1;

    cost = [];
    cnt = 1;
    cnvg = inf;
    while((cnt<iter)&&(cnvg>minLoss))
        [y_pred,a1,z1] = fwd2NN(theta_1,theta_2,X);
        dz2 = (y_pred'-Y);      %partial deriv. of loss wrt z2
        dtheta_2 = (a1*dz2)/length(Y);  %partial deriv. of z2 wrt a1 (=a1) * dz2
        
        da1 = dz2*theta_2(1,2:end); %removed the augmented first element (b)
        da1z1 = 1-a1(2:end,:).^2;       %partial deriv. of tanh
        dtheta_1 = (da1.*da1z1')'*X;

        theta_1 = theta_1-lr*dtheta_1;
        theta_2 = theta_2-lr*dtheta_2';
        loss = -sum(log(y_pred').*Y + log(1-y_pred').*(1-Y))/length(Y);
        
        if(rem(cnt,200)==0)
            plot(cost)
            pause(0.1)
        end
            cost = [cost loss];
        if(cnt>2)
            cnvg = abs(cost(end)-cost(end-1))/abs(cost(end-1));
        end
        cnt = cnt + 1;
    end
    
    Mdl.l1 = theta_1;
    Mdl.l2 = theta_2;

end


