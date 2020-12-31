function [trainAccuracy, trainLoss] = forwardPass (x_train, y_train, CNNtable, batch_size)

  trainAccuracy = 0; trainLoss = 0;
  for i = 1:batch_size
    x = reshape(x_train(i, :, :, :), size(x_train, 2), size(x_train, 3), size(x_train, 4));
    x = convolution(x, CNNtable, "layer1", activation = false);
    x = pooling(x);
    x = convolution(x, CNNtable, "layer2");
    x = pooling(x);
    x = convolution(x, CNNtable, "layer3");
    
    [~, y_predict_each] = max(x);
    if y_predict_each == y_train(i)
      trainAccuracy = trainAccuracy + 1;
    endif
    
    trainLoss = trainLoss + (y_predict_each - y_train(i)) ^ 2;
  endfor
  
  trainAccuracy = trainAccuracy / batch_size;
  
endfunction