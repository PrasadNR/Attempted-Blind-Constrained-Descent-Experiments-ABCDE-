function CNNtable = normalCNNfilters (lr, savedCNNtable)

  CNNtable = struct();
  
  CNNtable.layer1 =  savedCNNtable.layer1 + normrnd(0, lr, [5, 5, 16]);
  CNNtable.layer2 =  savedCNNtable.layer2 + normrnd(0, lr, [5, 5, 16]);
  CNNtable.layer3 =  savedCNNtable.layer3 + normrnd(0, lr, [5, 5, 10]);
  
endfunction