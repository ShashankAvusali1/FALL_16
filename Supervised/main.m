close all;
clc
clear
rng(100);
data = load('data\german.data-numeric.txt');
order = data(:,25) == 1;
class1 = data(order,:);
order = data(:,25) == 2;
class2 = data(order,:);

train = [class1(1:500,:); class2(1:100,:)];
test = [class1(500:700,:); class2(100:300,:)];

sample = random_oversample(train);

svm = svm_impl(sample(:,1:24),sample(:,25));
svm = fitPosterior(svm,sample(:,1:24),sample(:,25));
[labels,scores] = predict(svm,test(:,1:24));

