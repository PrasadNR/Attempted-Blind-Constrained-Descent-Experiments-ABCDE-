function CNNtable = randomCNNfilters ()

  CNNtable = struct();
  maxValue = 1;
  
  CNNtable.layer1 =  unifrnd(-maxValue, maxValue, [5, 5, 32]);
  CNNtable.layer2 =  unifrnd(-maxValue, maxValue, [5, 5, 32]);
  CNNtable.layer3 =  unifrnd(-maxValue, maxValue, [5, 5, 10]);
  
endfunction