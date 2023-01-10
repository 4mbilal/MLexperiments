function Mdl = fit3LayerNN(Xtrain,Ytrain,Xtest,Ytest,nc0,nc1,lr,iter,minLoss,lambda)
    Xtrain = [ones(size(Ytrain)),Xtrain];
    Xtest = [ones(size(Ytest)),Xtest];
    s = size(Xtrain);

    theta_0 = randn(nc0,s(2))*1;
    theta_1 = randn(nc1,nc0+1)*1;
    theta_2 = randn(1,nc1+1)*1;   %(output,input+1)
    cost = [];
    trainErr = [];
    testErr = [];
    cnt = 1;
    cnvg = inf;
%     figure
%     hold on
    while((cnt<iter)&&(cnvg>minLoss))
        [y_pred,a1,z1,a0,z0] = fwd3NN(theta_0,theta_1,theta_2,Xtrain);
        dL_dz2 = (y_pred'-Ytrain);
        dz2_dtheta_2 = a1;
        dL_dtheta_2 = (dz2_dtheta_2*dL_dz2)/length(Ytrain);
        theta_2 = theta_2-(lr)*dL_dtheta_2';
        theta_2 = theta_2-(lr*lambda/length(Ytrain));
        
        dz2_da1 = theta_2(1,2:end); %removed the augmented first element (b)
        dL_da1 = dL_dz2*dz2_da1;
        da1_dz1 = 1-a1(2:end,:).^2;
        dL_dz1 = (dL_da1.*da1_dz1')';
        dz1_dtheta_1 = a0';
        dL_dtheta_1 = dL_dz1*dz1_dtheta_1;
        theta_1 = theta_1-lr*dL_dtheta_1;
        theta_1 = theta_1-(lr*lambda/length(Ytrain));
        
        dz1_da0 = theta_1(:,2:end); %removed the augmented first element (b)
        dL_da0 = dL_dz1'*dz1_da0;
        da0_dz0 = 1-a0(2:end,:).^2;
        dL_dz0 = (dL_da0.*da0_dz0')';
        dz0_dtheta_0 = Xtrain;
        dL_dtheta_0 = dL_dz0*dz0_dtheta_0;
        theta_0 = theta_0-lr*dL_dtheta_0;
        theta_0 = theta_0-(lr*lambda/length(Ytrain));

        loss = -sum(log(y_pred').*Ytrain + log(1-y_pred').*(1-Ytrain))/length(Ytrain);
        loss = loss + lambda*(sum(sum(theta_0.^2))+sum(sum(theta_1'.^2))+sum(sum(theta_2'.^2)))/(2*length(Ytrain));
        cost = [cost loss];
        trainErr = [trainErr sum(double((y_pred'>0.5)~=Ytrain))/length(Ytrain)];
        [y_pred,a1,z1,a0,z0] = fwd3NN(theta_0,theta_1,theta_2,Xtest);
        testErr = [testErr sum(double((y_pred'>0.5)~=Ytest))/length(Ytest)];
        if(cnt>2)
            cnvg = abs(cost(end)-cost(end-1))/abs(cost(end-1));
        end
        subplot(1,2,1)
        plot(trainErr,'r')
        hold on
        plot(testErr,'g')
        legend('Training Error','Test Error');
        subplot(1,2,2)
        plot(cost,'b')
        legend('Training Loss');
        drawnow
%         ylim([0 2])
        pause(0.1)
        cnt = cnt + 1;
    end
    
    Mdl.l0 = theta_0;
    Mdl.l1 = theta_1;
    Mdl.l2 = theta_2;

end


