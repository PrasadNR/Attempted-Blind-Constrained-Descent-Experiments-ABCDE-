function CNNtable = randomCNNfilters ()

  CNNtable = struct();
  CNNtable.layer1 =  unifrnd(-1, 1, [5, 5, 6]);
  CNNtable.layer2 =  unifrnd(-1, 1, [5, 5, 16]);
  CNNtable.layer3 =  unifrnd(-1, 1, [5, 5, 120]);
  CNNtable.layer4 =  unifrnd(-1, 1, [1, 1, 84]);
  CNNtable.layer5 =  unifrnd(-1, 1, [1, 1, 10]);

endfunction