function [xTrain, yTrain] = shuffle (x_train, y_train)

  idx = randperm(size(x_train, 1));
  xTrain = x_train(idx, :, :, :); yTrain = y_train(idx);

endfunction