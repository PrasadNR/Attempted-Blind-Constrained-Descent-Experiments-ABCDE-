function CNNtable = uniformCNNfilters (lr, savedCNNtable)

  CNNtable = struct();
  
  CNNtable.layer1 =  savedCNNtable.layer1 + unifrnd(-lr, lr, [5, 5, 16]);
  CNNtable.layer2 =  savedCNNtable.layer2 + unifrnd(-lr, lr, [5, 5, 16]);
  CNNtable.layer3 =  savedCNNtable.layer3 + unifrnd(-lr, lr, [5, 5, 10]);
  
endfunction