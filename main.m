clc; close; clear;

dataFolder = "D:\\postCompletion\\research\\data";
addpath("helper");

cifar10 = load(fullfile(dataFolder, "cifar10.mat"));
cifar100 = load(fullfile(dataFolder, "cifar100.mat"));

x_train10 = cifar10.x_train; y_train10 = cifar10.y_train;
maxTrainAccuracy = 0; Nepochs = 100;
savedCNNtable = randomCNNfilters();

cifar10plot = zeros(1, Nepochs); cifar100plot = zeros(1, Nepochs);

tic;
for i = 1:Nepochs
  [x_train, y_train] = pickRandomTrainData(x_train10, y_train10, batch_size = 64);
  CNNtable = randomCNNfilters();
  trainAccuracy = forwardPass(x_train, y_train, CNNtable, batch_size);
  if trainAccuracy > maxTrainAccuracy
    maxTrainAccuracy = trainAccuracy;
    savedCNNtable = CNNtable;
  endif
  cifar10plot(i) = maxTrainAccuracy * 100;
end
toc;

save("outputs/cifar10CNNtable.mat", "savedCNNtable");

x_train100 = cifar100.x_train; y_train100 = cifar100.y_train;
maxTrainAccuracy = 0; tic;
for i = 1:Nepochs
  [x_train, y_train] = pickRandomTrainData(x_train100, y_train100, batch_size = 256);
  CNNtable = randomCNNfilters();
  trainAccuracy = forwardPass(x_train, y_train, CNNtable, batch_size);
  if trainAccuracy > maxTrainAccuracy
    maxTrainAccuracy = trainAccuracy;
    savedCNNtable = CNNtable;
  endif
  cifar100plot(i) = maxTrainAccuracy * 100;
end
toc;

save("outputs/cifar100CNNtable.mat", "savedCNNtable");

plot(cifar10plot);
hold on;
plot(cifar100plot);
hold off;
title("Plain Blind CNN Descent with uniform distribution");
xlabel("Epochs"); ylabel("Training Accuracy %");
legend("CIFAR-10", "CIFAR-100");
savefig("outputs/uniformDistribution.jpg");