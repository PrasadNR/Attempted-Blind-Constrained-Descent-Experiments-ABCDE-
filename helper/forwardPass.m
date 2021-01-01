function [trainAccuracy, trainLoss] = forwardPass (x_train, y_train, CNNtable, batch_size, numberOfClasses)

  trainAccuracy = 0; trainLoss = 0;
  for i = 1:batch_size
    x = reshape(x_train(i, :, :, :), size(x_train, 2), size(x_train, 3), size(x_train, 4));
    x = convolution(x, CNNtable, "layer1", activation = false);
    x = pooling(x);
    x = convolution(x, CNNtable, "layer2");
    x = pooling(x);
    x = convolution(x, CNNtable, "layer3", activation = false);
    predictions = reshape(x, numberOfClasses, 1);
    
    [~, y_predict_each] = max(predictions);
    if y_predict_each == y_train(i)
      trainAccuracy = trainAccuracy + 1;
    endif
    
    trainLoss = trainLoss + softmaxCEloss(y_train(i), predictions);
  endfor
  
  trainAccuracy = trainAccuracy / batch_size;
  
endfunction