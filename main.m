clc; close; clear;

dataFolder = "D:\\postCompletion\\research\\data";
cifar10 = load(fullfile(dataFolder, "cifar10.mat"));
cifar100 = load(fullfile(dataFolder, "cifar100.mat"));