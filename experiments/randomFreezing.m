clc; close; clear;

mainFolder = "D:\\postCompletion\\research";
dataFolder = fullfile(mainFolder, "data");
addpath(fullfile(mainFolder, "Accelerated-Blind-CNN-Descent-ABCD-", "helper"));

mnist = load(fullfile(dataFolder, "mnist.mat"));

x_train_MNIST = mnist.x_train / 255; y_train_MNIST = (mnist.y_train + 1)';
x_train_MNIST = padarray(x_train_MNIST, [0, 2, 2]);
savedTrainAccuracy = 0; Nepochs = 2; batch_size = 16;
savedCNNtable = randomCNNfilters(); CNNtable = randomCNNfilters();

[x_train, y_train] = shuffle(x_train_MNIST, y_train_MNIST);
count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
  endfor
endfor
MNISTplot = zeros(1, count);

tic; minLoss = Inf; savedTrainAccuracy = 0; count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
    idx = (j * batch_size):(batch_size * (j + 1));
    x_train_batch = x_train(idx, :, :, :); y_train_batch = y_train(idx);
    CNNtable = normalCNNfilters(lr = 0.001, savedCNNtable);
    CNNtable = randomFreeze (freezeFactor = 0.75, CNNtable, savedCNNtable);
    [trainAccuracy, trainLoss] = forwardPass(x_train_batch, y_train_batch, CNNtable, batch_size, numberOfClasses);
    if trainLoss < minLoss
      minLoss = trainLoss;
      savedTrainAccuracy = trainAccuracy;
      savedCNNtable = CNNtable;
    endif
    MNISTplot(count) = savedTrainAccuracy * 100;
  endfor
end
toc;

plot(MNISTplot);