function newCNNtable = layerByLayer (epoch, CNNtable, savedCNNtable)

  newCNNtable = savedCNNtable;
  Nlayers = numfields(savedCNNtable);
  layer = strcat("layer", num2str(mod(epoch, Nlayers) + 1));
  activeLayer = getfield(CNNtable, layer);
  setfield(newCNNtable, layer, activeLayer);

endfunction