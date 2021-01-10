function loss = softmaxCEloss (trueLabel, predictions)

  stdPredictions = predictions / std(predictions);
  shiftedPredictions = stdPredictions - max(stdPredictions);
  expPredictions = exp(shiftedPredictions);
  softmaxPredictions = expPredictions / sum(expPredictions);
  logPredictions = log(softmaxPredictions);
  loss = -logPredictions(trueLabel);
  
endfunction