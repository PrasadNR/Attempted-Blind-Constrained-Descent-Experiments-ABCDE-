clc; close; clear;

dataFolder = "D:\\postCompletion\\research\\data";
addpath("helper");

mnist = load(fullfile(dataFolder, "mnist.mat"));

x_train_MNIST = mnist.x_train; y_train_MNIST = mnist.y_train + 1;
x_train_MNIST = padarray(x_train_MNIST, [0, 2, 2]);
maxTrainAccuracy = 0; Nepochs = 1000;
savedCNNtable = randomCNNfilters(); CNNtable = randomCNNfilters();

MNISTplot = zeros(1, Nepochs);

tic;
for i = 1:Nepochs
  [x_train, y_train] = pickRandomTrainData(x_train_MNIST, y_train_MNIST, batch_size = 16);
  CNNtable = normalCNNfilters(lr = 0.001, savedCNNtable);
  CNNtable = layerByLayer (i, CNNtable, savedCNNtable);
  trainAccuracy = forwardPass(x_train, y_train, CNNtable, batch_size);
  if trainAccuracy > maxTrainAccuracy
    maxTrainAccuracy = trainAccuracy;
    savedCNNtable = CNNtable;
  endif
  MNISTplot(i) = maxTrainAccuracy * 100;
end
toc;

plot(MNISTplot);