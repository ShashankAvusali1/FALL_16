function [ sample ] = random_oversample( data )
%   Perform random over sampling of minority class
    order = data(:,25) == 1;
    class1 = data(order,:);
    order = data(:,25) == 2;
    class2 = data(order,:);
    [majority_count,~] = size(class1);
    sample = datasample(class2,majority_count,'Replace',true);
    sample = [class1; sample];
end

