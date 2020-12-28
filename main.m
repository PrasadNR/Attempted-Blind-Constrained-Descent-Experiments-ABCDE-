clc; close; clear;

dataFolder = "D:\\postCompletion\\research\\data";
cifar10 = load(fullfile(dataFolder, "cifar10.mat"));
cifar100 = load(fullfile(dataFolder, "cifar100.mat"));

x_train10 = cifar10.x_train; y_train10 = cifar10.y_train;
[x_train, y_train] = pickRandomTrainData(x_train10, y_train10, batch_size = 64);
CNNtable = initialiseNetwork();
tic;
trainAccuracy = forwardPass(x_train, CNNtable, batch_size);
toc;