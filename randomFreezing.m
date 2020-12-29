clc; close; clear;

dataFolder = "D:\\postCompletion\\research\\data";
addpath("helper");

mnist = load(fullfile(dataFolder, "mnist.mat"));

x_train10 = mnist.x_train; y_train10 = mnist.y_train;
x_train10 = x_train10 / 255;
x_train10 = padarray(x_train10, [0, 2, 2]);
maxTrainAccuracy = 0; Nepochs = 100;
savedCNNtable = randomCNNfilters(); CNNtable = randomCNNfilters();

cifar10plot = zeros(1, Nepochs);

tic;
for i = 1:Nepochs
  [x_train, y_train] = pickRandomTrainData(x_train10, y_train10, batch_size = 64);
  #CNNtable = randomFreeze (freezeFactor = 0.9, CNNtable, savedCNNtable);
  CNNtable = randomCNNfilters();
  trainAccuracy = forwardPass(x_train, y_train, CNNtable, batch_size);
  if trainAccuracy > maxTrainAccuracy
    maxTrainAccuracy = trainAccuracy;
    savedCNNtable = CNNtable;
  #else
    #CNNtable = randomCNNfilters();
  endif
  cifar10plot(i) = maxTrainAccuracy * 100;
end
toc;

plot(cifar10plot);