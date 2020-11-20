clear; close all; clc;

%% preparing dataset

Data=dlmread('Data.txt');
%%
X = Data(:,2:7); 
y = Data(:,8);

X_train = X(1:20,:);
y_train = y(1:20,:);

X_test = X(21:60,:);

%% CV partition

c = cvpartition(y_train,'k',5);
%% feature selection

opts = statset('display','iter');
classf = @(train_data, train_labels, test_data, test_labels)...
    sum(predict(fitcsvm(train_data, train_labels,'KernelFunction','rbf'), test_data) ~= test_labels);

[fs, history] = sequentialfs(classf, X_train, y_train, 'cv', c, 'options', opts,'nfeatures',2);
%% Best hyperparameter

X_train_w_best_feature = X_train(:,fs);

Md1 = fitcsvm(X_train_w_best_feature,y_train,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
      'expected-improvement-plus','ShowPlots',true)); 


%% Final test with test set
X_test_w_best_feature = X_test(:,fs);
predict(Md1,X_test_w_best_feature);



