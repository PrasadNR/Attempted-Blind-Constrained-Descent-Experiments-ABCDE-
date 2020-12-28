function trainAccuracy = forwardPass (x_train, CNNtable, batch_size)

  for i = 1:batch_size
    x = reshape(x_train(i, :, :, :), size(x_train, 2), size(x_train, 3), size(x_train, 4));
    x = convolution(x, CNNtable, "layer1");
    x = pooling(x);
    x = convolution(x, CNNtable, "layer2");
    x = pooling(x);
    x = convolution(x, CNNtable, "layer3");
    x = convolution(x, CNNtable, "layer4");
    x = convolution(x, CNNtable, "layer5");
    
    [~, y_predictEach] = max(x);
  endfor
  trainAccuracy = 0;
  
endfunction