function [ model ] = svm_impl( traindata,labels )
%   Trains a svm model on given dataset
%     x = -5:12;
%     C = 2.^x;
%     x = -12:5;
%     sigma = 2.^x;
    model = fitcsvm(traindata,labels,'KernelScale','auto','KernelFunction','rbf',...
        'Solver','SMO','Cost',[0,1;5,0],'CrossVal','on',...
        'Standardize',true,'OptimizeHyperparameters','auto','Verbose',1,...
        'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',50));
%           'BoxConstraint',1, ...  
    
end

