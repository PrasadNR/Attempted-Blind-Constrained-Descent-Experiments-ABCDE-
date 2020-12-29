function featureMaps = convolution (inputImageChannels, CNNtable, ID, activation = true)

  kernels2D = getfield(CNNtable, ID);

  A = sum(inputImageChannels, 3);
  B = kernels2D(:, :, 1);
  C = conv2(A, B, SHAPE = "valid");
  featureMaps = zeros(size(C, 1), size(C, 2), size(kernels2D, 3));
  featureMaps(:, :, 1) = C;
  
  if size(kernels2D, 3) > 1
    for i = 2:size(kernels2D, 3)
      B = kernels2D(:, :, i);
      featureMaps(:, :, i) = conv2(A, B, SHAPE = "valid");
    endfor
  endif
  
  if activation
    featureMaps = max(featureMaps, 0);
  endif

endfunction