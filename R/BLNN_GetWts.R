#' @title Extract the vector of network weights from a BLNN object
#' @name BLNN_GetWts
#' @description Allows the user to extract all weight values from a BLNN network.
#' object.
#' @param Network A BLNN network object
#' @return A vector containing all weight parameters from the network. The weights are sorted
#' by imput, having all weights from covariate one that connect to hidden node 1:n. Each
#' subsequent covariates weights for layer one are saved in this order followed by
#' the weights associated with the bias term for layer one. Layer two weights follow
#' a similar format having each hidden units weighes followed by bias weights.
#' @export


BLNN_GetWts<-function(Network){

  wts<-.VecNetWts(Network)

  return(wts)
}
