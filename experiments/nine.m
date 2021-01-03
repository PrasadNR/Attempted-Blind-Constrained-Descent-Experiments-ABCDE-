clc; close; clear;

mainFolder = "D:\\postCompletion\\research";
dataFolder = fullfile(mainFolder, "data");
addpath(fullfile(mainFolder, "Accelerated-Blind-CNN-Descent-ABCD-", "helper"));

cifar10 = load(fullfile(dataFolder, "cifar10.mat"));
mnist = load(fullfile(dataFolder, "mnist.mat"));
numberOfClasses = 10;

x_train10 = cifar10.x_train / 255; y_train10 = (cifar10.y_train + 1)';
Nepochs = 40; batch_size = 16;
[x_train, y_train] = shuffle(x_train10, y_train10);

freezeFactor = 0.75;
lr = 0.001; count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
  endfor
endfor
cifar10TrainAccuracyPlot = zeros(1, count); cifar10TrainLossPlot = zeros(1, count);
MNISTtrainAccuracyPlot = zeros(1, count); MNISTtrainLossPlot = zeros(1, count);

saveStruct = struct();

CNNtable = initialNormalCNNfilters(lr); savedCNNtable = initialNormalCNNfilters(lr);
tic; minLoss = Inf; savedTrainAccuracy = 0; count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
    idx = (j * batch_size):(batch_size * (j + 1));
    x_train_batch = x_train(idx, :, :, :); y_train_batch = y_train(idx);
    CNNtable = normalCNNfilters(lr, savedCNNtable);
    CNNtable = randomFreeze(freezeFactor, CNNtable, savedCNNtable);
    [trainAccuracy, trainLoss] = forwardPass(x_train_batch, y_train_batch, CNNtable, batch_size, numberOfClasses);
    if trainLoss < minLoss
      minLoss = trainLoss;
      savedTrainAccuracy = trainAccuracy;
      savedCNNtable = CNNtable;
    endif
    cifar10TrainAccuracyPlot(count) = savedTrainAccuracy * 100;
    cifar10TrainLossPlot(count) = minLoss;
  endfor
endfor
toc;

saveStruct.cifar10TrainAccuracyPlot = cifar10TrainAccuracyPlot;
saveStruct.cifar10TrainLossPlot = cifar10TrainLossPlot;
saveStruct.cifar10CNN = savedCNNtable;

x_train_mnist = mnist.x_train / 255; y_train_mnist = (mnist.y_train + 1)';
x_train_mnist = padarray(x_train_mnist, [0, 2, 2]);

CNNtable = initialNormalCNNfilters(lr); savedCNNtable = initialNormalCNNfilters(lr);
tic; minLoss = Inf; savedTrainAccuracy = 0; count = 0;
for i = 1:Nepochs
  for j = 1:(floor(size(x_train, 1) / batch_size) - 1)
    count = count + 1;
    idx = (j * batch_size):(batch_size * (j + 1));
    x_train_batch = x_train(idx, :, :, :); y_train_batch = y_train(idx);
    CNNtable = normalCNNfilters(lr, savedCNNtable);
    CNNtable = randomFreeze(freezeFactor, CNNtable, savedCNNtable);
    [trainAccuracy, trainLoss] = forwardPass(x_train_batch, y_train_batch, CNNtable, batch_size, numberOfClasses);
    if trainLoss < minLoss
      minLoss = trainLoss;
      savedTrainAccuracy = trainAccuracy;
      savedCNNtable = CNNtable;
    endif
    MNISTtrainAccuracyPlot(count) = savedTrainAccuracy * 100;
    MNISTtrainLossPlot(count) = minLoss;
  endfor
endfor
toc;

saveStruct.MNISTtrainAccuracyPlot = MNISTtrainAccuracyPlot;
saveStruct.MNISTtrainLossPlot = MNISTtrainLossPlot;
saveStruct.MNIST_CNN = savedCNNtable;

save(fullfile(mainFolder, "Accelerated-Blind-CNN-Descent-ABCD-", "outputs", "nine.mat"), "saveStruct");