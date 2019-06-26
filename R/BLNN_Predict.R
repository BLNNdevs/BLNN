#' @title Calculate predicted values and errors for a BLNN object
#' @name BLNN_Predict
#' @description Allows the user to calculate predicted values from input data with the
#' option of providing network and fitted errors if known y values are supplied.
#' @param Network A BLNN network object
#' @param x A set of input data
#' @param y A set of target response values. Default is \code{NULL}
#' @return A vector or matrix of predicted responses. If y is given, returns a list containing
#' \describe{
#' \item{Errors}{Calculated network error}
#' \item{Difference}{A vector of differences y-predicted}
#' \item{Fitted_Values}{A vector or matrix of predicted responses}
#' }
#' @export

BLNN_Predict<-function(Network, x, y=NULL){

  forwarded<-ffwd(Network, x)


  if (Network$outF=="softmax"){
    return(max.col(forwarded$trainout))
  } else{ if(is.null(y)==TRUE){
    return(forwarded$trainout)
  } else{
    Err<-.NetErr(y, Network, x, sep = TRUE)
    diff<-(y-forwarded$trainout)
    return(list(Errors=Err, Difference=diff, Fitted_Values=forwarded$trainout))
  }
}}
