function [ model ] = svm_impl( traindata,labels )
%   Trains a svm model on given dataset
    c = cvpartition(size(traindata,1),'LeaveOut');
    sigma = optimizableVariable('sigma',[1e-5,1e5],'Transform','log');
    box = optimizableVariable('box',[1e-5,1e5],'Transform','log');
    model = fitcsvm(traindata,labels,'KernelFunction','rbf','KernelScale','auto','ClassNames',[1,2]);
    
end

