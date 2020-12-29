function CNNtable = randomFreeze (freezeFactor = 0.5, CNNtable, savedCNNtable)

  outputTable = CNNtable;

  for i = 1:numfields(savedCNNtable)
    layer = strcat("layer", num2str(i));
    savedCNNlayer = getfield(savedCNNtable, layer);
    frozenCNNlayer = getfield(CNNtable, layer);
    
    imax = size(frozenCNNlayer, 3); N = round(freezeFactor * imax);
    idx = randi(imax, 1, N);
    
    frozenCNNlayer(:, :, idx) = savedCNNlayer(:, :, idx);
    CNNtable = setfield(CNNtable, layer, frozenCNNlayer);
  endfor
  
endfunction