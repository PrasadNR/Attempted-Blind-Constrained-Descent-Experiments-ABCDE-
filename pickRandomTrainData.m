function [xTrain, yTrain] = pickRandomTrainData (x_train, y_train, batch_size)

  idx = randi(size(x_train, 1), 1, batch_size);
  xTrain = x_train(idx, :, :, :); yTrain = y_train(idx);

endfunction