function trainAccuracy = forwardPass (x_train, y_train, CNNtable, batch_size)

  trainAccuracy = 0;
  for i = 1:batch_size
    x = reshape(x_train(i, :, :, :), size(x_train, 2), size(x_train, 3), size(x_train, 4));
    x = convolution(x, CNNtable, "layer1", activation = false);
    x = pooling(x);
    x = convolution(x, CNNtable, "layer2");
    x = pooling(x);
    x = convolution(x, CNNtable, "layer3", activation = false);
    x = convolution(x, CNNtable, "layer4", activation = false);
    x = convolution(x, CNNtable, "layer5");
    
    [~, y_predict_each] = max(x);
    if y_predict_each == y_train(i)
      trainAccuracy = trainAccuracy + 1;
    endif
  endfor
  
  trainAccuracy = trainAccuracy / batch_size;
  
endfunction