function [ sample ] = random_undersample( data )
%   Perform random under sampling of majority class
    order = data(:,25) == 1;
    class1 = data(order,:);
    order = data(:,25) == 2;
    class2 = data(order,:);
    [minority_count,~] = size(class2);
    sample = datasample(class1,minority_count,'Replace',false);
    sample = [sample; class2];
end

