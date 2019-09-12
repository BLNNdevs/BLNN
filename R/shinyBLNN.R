#' @title Launch ShinyStan diagnosis
#' @name launch_shinyBLNN
#' @description lanuch the shinyStan tool fo diagnosing the posterior samples using the NUTS algorithm.
#' @param fit the BLNN_Train object.
#'
#' @export

launch_shinyBLNN <- function(fit){
  shinystan::launch_shinystan(.as.shinyBLNN(fit))
}



