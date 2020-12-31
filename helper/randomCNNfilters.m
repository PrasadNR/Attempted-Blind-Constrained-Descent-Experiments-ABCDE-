function CNNtable = randomCNNfilters ()

  CNNtable = struct();
  
  CNNtable.layer1 =  unifrnd(-1, 1, [5, 5, 32]);
  CNNtable.layer2 =  unifrnd(-1, 1, [5, 5, 32]);
  CNNtable.layer3 =  unifrnd(-1, 1, [5, 5, 10]);
  
endfunction