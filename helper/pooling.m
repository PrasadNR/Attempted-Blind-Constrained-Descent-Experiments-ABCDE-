function featureMaps = pooling (inputMaps, maxpoolHeight = 2, maxpoolWidth = 2)

  outputSizeH = ceil(size(inputMaps, 1) / maxpoolHeight);
  outputSizeW = ceil(size(inputMaps, 2) / maxpoolWidth);
  featureMaps = zeros(outputSizeH, outputSizeW, size(inputMaps, 3));
  
##  maxpool = @(block_struct) max(block_struct(:));
##  
  for i = 1:size(inputMaps, 3)
    ## featureMaps(:, :, i) = blockproc(inputMaps(:, :, i), [maxpoolHeight, maxpoolWidth], maxpool);
    zerosFilter = zeros(outputSizeH * maxpoolHeight, outputSizeW * maxpoolWidth);
    eachFilter = inputMaps(:, :, i);
    zerosFilter(1:size(eachFilter, 1), 1:size(eachFilter, 2)) = eachFilter;
    outputFilter = zeros(outputSizeH, outputSizeW, maxpoolHeight * maxpoolWidth);
    
    l = 0;
    for j = 1:maxpoolHeight
      for k = 1:maxpoolWidth
        l = l + 1;
        outputFilter(:, :, l) = zerosFilter(j:maxpoolHeight:end, k:maxpoolWidth:end);
      endfor
    endfor
    
    featureMaps(:, :, i) = max(outputFilter, [], 3);
  endfor

##  featureMaps = inputMaps(1:maxpoolHeight:end, 1:maxpoolWidth:end, :);

endfunction