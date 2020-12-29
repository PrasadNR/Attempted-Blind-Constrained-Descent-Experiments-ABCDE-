clc; close; clear;

dataFolder = "D:\\postCompletion\\research\\data";
cifar10 = load(fullfile(dataFolder, "cifar10.mat"));
cifar100 = load(fullfile(dataFolder, "cifar100.mat"));

x_train10 = cifar10.x_train; y_train10 = cifar10.y_train;
maxTrainAccuracy = 0; Nepochs = 100;
tic;
for i = 1:Nepochs
  [x_train, y_train] = pickRandomTrainData(x_train10, y_train10, batch_size = 64);
  CNNtable = initialiseNetwork();
  trainAccuracy = forwardPass(x_train, y_train, CNNtable, batch_size);
  if trainAccuracy > maxTrainAccuracy
    maxTrainAccuracy = trainAccuracy;
    savedCNNtable = CNNtable;
  endif
end
toc;

save("savedCNNtable.mat", "savedCNNtable");