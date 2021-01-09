function CNNtable = randomFreeze (freezeFactor = 0.5, CNNtable, savedCNNtable)

  for i = 1:numfields(savedCNNtable)
    layer = strcat("layer", num2str(i));
    savedCNNlayer = getfield(savedCNNtable, layer);
    frozenCNNfilters = getfield(CNNtable, layer);
    
    imax = size(frozenCNNfilters, 3); N = round(freezeFactor * imax);
    idx = randi(imax, 1, N);
    
    frozenCNNfilters(:, :, idx) = savedCNNlayer(:, :, idx);
    CNNtable = setfield(CNNtable, layer, frozenCNNfilters);
  endfor
  
endfunction