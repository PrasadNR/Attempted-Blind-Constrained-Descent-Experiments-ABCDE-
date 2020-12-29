function featureMaps = pooling (inputMaps, maxpoolHeight = 2, maxpoolWidth = 2)

##  outputSizeH = ceil(size(inputMaps, 1) / maxpoolHeight);
##  outputSizeW = ceil(size(inputMaps, 2) / maxpoolWidth);
##  featureMaps = zeros(outputSizeH, outputSizeW, size(inputMaps, 3));
##  
##  maxpool = @(block_struct) max(block_struct(:));
  
##  for i = 1:size(inputMaps, 3)
##    featureMaps(:, :, i) = blockproc(inputMaps(:, :, i), [maxpoolHeight, maxpoolWidth], maxpool);
##  endfor

  featureMaps = inputMaps(1:maxpoolHeight:end, 1:maxpoolWidth:end, :);

endfunction