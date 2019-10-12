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

BLNN_Predict <- function(Network, x, y = NULL, fit = NULL) {
  if (is.null(fit)== TRUE) {
    forwarded <- .ffwd(Network, x)


    if (Network$outF == "softmax") {
      return(max.col(forwarded$trainout))
    } else{
      if (is.null(y) == TRUE) {
        return(forwarded$trainout)
      } else{
        Err <- .NetErr(y, Network, x, sep = TRUE)
        diff <- (y - forwarded$trainout)
        return(list(
          Errors = Err,
          Difference = diff,
          Predicted_Values = forwarded$trainout
        ))
      }
    }
  } else{
        fil.net <- function(w){
          nettemp <- BLNN:::.UpNetWts(w,Network)
          out <- BLNN_Predict(nettemp,x)
          return(out)
        }
        ### Collecting all accepted samples from fit object
        d1 <-NULL
        chains <-length(fit$sampler_params)
        for(k in 1:chains){
          m <- fit$samples[,k,]
          if(fit$algorithm=="HMC"){
          m1 <-as.data.frame(cbind(m,acc = fit$sampler_params[[k]][,5]))
          }else{m1 <-as.data.frame(cbind(m,acc = fit$sampler_params[[k]][,7]))}
          if(chains == 1){
            d1 <- m1 %>% dplyr::filter(acc == 1) %>% dplyr::select(-Er__,-acc)
          }else{
            d1 <- rbind(d1,m1 %>% dplyr::filter(acc == 1) %>% dplyr::select(-Er__,-acc))
          }
        }
        pred <- apply(d1,1,fil.net)
        if (Network$outF == "softmax"){
          return(apply(pred,1,function(x){as.numeric(names(table(x))[table(x)==max(table(x))])}))
        }else{
          return(colMeans(t(pred)))
        }

  }
}
