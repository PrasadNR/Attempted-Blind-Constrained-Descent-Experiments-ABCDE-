clc; close; clear;

mainFolder = "D:\\postCompletion\\research";
dataFolder = fullfile(mainFolder, "data");
addpath(fullfile(mainFolder, "Accelerated-Blind-CNN-Descent-ABCD-", "helper"));

cifar10 = load(fullfile(dataFolder, "cifar10.mat"));
mnist = load(fullfile(dataFolder, "mnist.mat"));
numberOfClasses = 10; batch_size = 10000;

x_test10 = cifar10.x_test / 255; y_test10 = (cifar10.y_test + 1)';

MATfile = input("Saved CNN struct MAT file name: ");
CNNstruct = load(fullfile(mainFolder, "Accelerated-Blind-CNN-Descent-ABCD-", "outputs", MATfile));
CNNtable = CNNstruct.saveStruct.cifar10CNN;
[CIFAR10testAccuracy, testLoss] = forwardPass(x_test10, y_test10, CNNtable, batch_size, numberOfClasses);
display(CIFAR10testAccuracy);

x_train_mnist = mnist.x_test / 255; y_train_mnist = (mnist.y_test + 1)';
x_train_mnist = padarray(x_train_mnist, [0, 2, 2]);
CNNtable = CNNstruct.saveStruct.MNIST_CNN;
[MNISTtestAccuracy, testLoss] = forwardPass(x_train_mnist, y_train_mnist, CNNtable, batch_size, numberOfClasses);
display(MNISTtestAccuracy);