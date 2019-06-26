
launch_shinyBLNN <- function(fit){
  shinystan::launch_shinystan(.as.shinyBLNN(fit))
}


.as.shinyBLNN <-function(fit){
  if (fit$algorithm == "NUTS") {
    sso <- with(fit, shinystan::as.shinystan(samples, warmup = warmup,
                                             max_treedepth = max_treedepth, sampler_params = sampler_params,
                                             algorithm = "NUTS", model_name = "BLNN"))
  }
  else if (fit$algorithm == "HMC") {
    sso <- with(fit, shinystan::as.shinystan(samples, warmup = warmup,
                                             sampler_params = sampler_params,
                                             algorithm = "HMC",
                                             model_name = "BLNN"))
  }

  return(invisible(sso))

}
