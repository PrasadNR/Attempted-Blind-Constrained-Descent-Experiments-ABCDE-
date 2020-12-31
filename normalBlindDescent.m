clc; close; clear;

dataFolder = "D:\\postCompletion\\research\\data";
addpath("helper");

cifar10 = load(fullfile(dataFolder, "cifar10.mat"));
mnist = load(fullfile(dataFolder, "mnist.mat"));

x_train10 = cifar10.x_train; y_train10 = cifar10.y_train + 1;
Nepochs = 2; batch_size = 16;
savedCNNtable = randomCNNfilters();
[x_train, y_train] = shuffle(x_train10, y_train10);

CNNtable = randomCNNfilters(); lr = 0.001;
count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
  endfor
endfor
cifar10plot = zeros(1, count); MNISTplot = zeros(1, count);

tic; minLoss = Inf; savedTrainAccuracy = 0; count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
    idx = (j * batch_size):(batch_size * (j + 1));
    x_train_batch = x_train(idx, :, :, :); y_train_batch = y_train(idx);
    CNNtable = normalCNNfilters(lr = 0.001, savedCNNtable);
    [trainAccuracy, trainLoss] = forwardPass(x_train_batch, y_train_batch, CNNtable, batch_size);
    if trainLoss < minLoss
      minLoss = trainLoss;
      savedTrainAccuracy = trainAccuracy;
      savedCNNtable = CNNtable;
    endif
    cifar10plot(count) = savedTrainAccuracy * 100;
  endfor
endfor
toc;

x_train_mnist = mnist.x_train; y_train_mnist = mnist.y_train + 1;
x_train_mnist = padarray(x_train_mnist, [0, 2, 2]);
CNNtable = randomCNNfilters();
tic; minLoss = Inf; savedTrainAccuracy = 0; count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
    idx = (j * batch_size):(batch_size * (j + 1));
    x_train_batch = x_train(idx, :, :, :); y_train_batch = y_train(idx);
    CNNtable = normalCNNfilters(lr = 0.001, savedCNNtable);
    trainAccuracy = forwardPass(x_train_batch, y_train_batch, CNNtable, batch_size);
    if trainLoss < minLoss
      minLoss = trainLoss;
      savedTrainAccuracy = trainAccuracy;
      savedCNNtable = CNNtable;
    endif
    MNISTplot(count) = savedTrainAccuracy * 100;
  endfor
endfor
toc;

plot(cifar10plot);
hold on;
plot(MNISTplot);
hold off;
title("Blind CNN Descent with uniform distribution");
xlabel("Batches"); ylabel("Training Accuracy %");
legend("CIFAR-10", "MNIST");