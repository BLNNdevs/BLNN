#' @title Calculate predicted values and errors for a BLNN object
#' @name BLNN_Update
#' @description Allows the user to update the network object after examining the samples.
#' If the samples drawn passed the diagnostic checks then we need to update the network object with the average value of the accetped samples after warmup.
#' @param Network A BLNN network object
#' @param fit The fitted object returned by \code{\link{BLNN_train}}.
#' @param index The starting index used to compute the average estimated value of the newtork weights. The default is warmup+1
#' @return The Network object with trained weights.
#'
#' @export

BLNN_Update <- function(NET, fit, index=NULL){
  sam <- fit$samples
  chains <- dim(sam)[2]
  ite <- dim(sam)[1]
  l <- dim(sam)[3]
  if(is.null(index)) index = fit$warmup+1
  if(chains == 1){
    wts.up <- colMeans(sam[index:ite,,])[-l]
  }else {
    wts.up <- colMeans(colMeans(sam[index:ite,,]))[-l]
  }

  Net.up <- .UpNetWts(wts.up,NET)
  return(Net.up)

}
