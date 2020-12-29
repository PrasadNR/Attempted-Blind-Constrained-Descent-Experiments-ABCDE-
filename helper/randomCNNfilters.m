function CNNtable = randomCNNfilters ()

  CNNtable = struct();
  maxValue = 1;
  CNNtable.layer1 =  unifrnd(-maxValue, maxValue, [5, 5, 6]);
  CNNtable.layer2 =  unifrnd(-maxValue, maxValue, [5, 5, 16]);
  CNNtable.layer3 =  unifrnd(-maxValue, maxValue, [5, 5, 120]);
  CNNtable.layer4 =  unifrnd(-maxValue, maxValue, [1, 1, 84]);
  CNNtable.layer5 =  unifrnd(-maxValue, maxValue, [1, 1, 10]);

endfunction