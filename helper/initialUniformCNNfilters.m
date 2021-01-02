function CNNtable = initialUniformCNNfilters (lr)

  CNNtable = struct();
  
  CNNtable.layer1 =  unifrnd(-lr, lr, [5, 5, 16]);
  CNNtable.layer2 =  unifrnd(-lr, lr, [5, 5, 16]);
  CNNtable.layer3 =  unifrnd(-lr, lr, [5, 5, 10]);
  
endfunction