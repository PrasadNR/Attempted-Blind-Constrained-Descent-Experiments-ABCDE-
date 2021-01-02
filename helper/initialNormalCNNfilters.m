function CNNtable = initialNormalCNNfilters (lr)

  CNNtable = struct();
  
  CNNtable.layer1 =  normrnd(0, lr, [5, 5, 16]);
  CNNtable.layer2 =  normrnd(0, lr, [5, 5, 16]);
  CNNtable.layer3 =  normrnd(0, lr, [5, 5, 10]);
  
endfunction