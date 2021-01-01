function loss = softmaxCEloss (trueLabel, predictions)

  stdPredictions = std(predictions);
  shiftedPredictions = stdPredictions - max(stdPredictions);
  expPredictions = exp(shiftedPredictions);
  softmaxPredictions = expPredictions / sum(expPredictions);
  logPredictions = log(predictions);
  loss = -logPredictions(trueLabel);
  
endfunction