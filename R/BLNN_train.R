#' @title Train a BLNN object
#' @name BLNN_Train
#' @description This function allow the user to train the BLNN object. The user can choose
#' from three training algorithms. "NUTS" and "HMC" for Bayesian training method, see references for detailed description. The third option is "BFGS" for
#' quasi-Newton method which is done through optim function.
#'
#' @param NET the BLNN object which is created using \code{\link{BLNN_Build}}.
#'
#' @param x A matrix or data frame of covariates. it is preferred that continuous variables are scaled before training.
#' @param y response or target values. A vector for one unit in the output layer, or a matrix/dataframe for more than one unit in the output layer.
#' @param iter is the number of samples to draw from the Posterior distribution. In case of "BFGS" algorithm it is the number of iterations.
#' @param init A list of vectors containing the initial parameters values or a function. It is strongly recommended to have a different vector for each chain
#' @param chains Number of chains to run. Needed for Bayesian training only.
#' @param seeds A vector of seeds. One for each chain.
#' @param warmup The number of warmup iterations/samples. Default is half the number of iter.
#' @param thin The thinning rate to apply to samples
#' @param parallel A boolean value to check whether to use Parallel cores or not. Snowfall package is needed if TRUE.
#' @param cores Number of cores to be used if parallel is TRUE.
#' @param algorithm choose one algorithm from three c("NUTS", "HMC", "BFGS"). NUTS for the NO-U-Turn algorithm, HMC for Hamiltonian Markov chain sampler.
#' See references below for detailed descriptions of each algorithm. The BFGS for quasi-Newton algorithm.
#' @param evidence A boolean value to use the evidence procedure for re-estimating the Hyper-parameters.
#' @param ev.x matrix/dataframe of covariates to be used in evidence procedure. Prefered to be historical data or part of the current training data. If left blank while evidence is TRUE, x will be used.
#' @param ev.y vector/matrix of targets to be used in evidence procedure. If left blank while evidence is TRUE, y will be used.
#' @param ev.iter number of iterations in evidence procedure, see references for more detials. Default is set to 1.
#' @param control A list containing several control arguments needed for tunning NUTS and HMC. These arguments are:
#' \itemize{
#' \item{adapt_delta: }{ The target acceptance rate. Default is \code{0.8}, for HMC preferred is \code{0.65}.}
#' \item{momentum.mass: }{ A vector of the momentum variance, default is \code{1}.}
#' \item{stepsize: }{ The stepsize to be used for NUTS algorithm. If \code{NULL} it will be adapted during warmup.}
#' \item{useDA: }{ Whether dual averaging for adapting stepsize is used or not. Default is \code{TRUE}.}
#' \item{gamma: }{ One of DA arguments, double, positive, defaults to \code{2}.}
#' \item{t0: }{ One of DA arguments, double, positive, defaults to \code{10}.}
#' \item{kappa: }{ One of DA arguments, double, positive, defaults to \code{0.75}.}
#' \item{metric: }{ The mass metric to use. Options are: "unit" for a unit diagonal matrix; \code{NULL} to estimate a diagonal matrix during warmup;
#' a matrix to be used directly (in untransformed space)}
#' \item{adapt_mass: }{ Whether adaptation of mass matrix is turned
#'   on. Currently only allowed for diagonal metric.}
#' \item{w1: }{ integer, positive, defaults to 75.}
#' \item{w2: }{ integer, positive, defaults to 50.}
#' \item{w3: }{ integer, positive, defaults to 25.}
#' In addition one argument used only for NUTS:
#' \item{max_treedepth: }{ integer, positive, defaults to 10}
#' For HMC algorithm we can also set:
#' \item{Lambda: }{ Simulation length of one trajectory, double,\code{[0,1]}}.
#' }
#'
#' @param display Help track the sampler algorithm by displaying several results. Value \code{0} display nothing, \code{1} display the
#' neural network error after each iteration. \code{2} will display the stepsize and number of leapfrog steps during and after warmup for each iteration.
#' \code{3} In addition to error function,stepsize, and leapfrog steps it will display the old and new energy for each iteration.
#'
#' @references
#' \itemize{ \item{Neal, R. M. (2011). MCMC using Hamiltonian
#'   dynamics. Handbook of Markov Chain Monte Carlo.}  \item{Hoffman and
#'   Gelman (2014). The No-U-Turn sampler: Adaptively setting path lengths
#'   in Hamiltonian Monte Carlo. J. Mach. Learn. Res.  15:1593-1623.}  }
#'
#'
#'
#'
#' @export
#'


BLNN_Train <-
  function(NET,
           x,
           y,
           iter = 2000,
           init=NULL,
           chains = 1,
           seeds = NULL,
           warmup = floor(iter / 2),
           thin = 1,
           parallel = FALSE,
           cores = NULL,
           algorithm = c("NUTS", "HMC", "BFGS"),
           evidence = FALSE,
           ev.x = NULL,
           ev.y = NULL,
           ev.iter = 1,
           control = NULL,
           display = 0,
           path=getwd(),
           ...) {
    if (is.null(x) | is.null(y)) {
      stop("Make sure network's input and target are assigned")
    }

    Actual <- y
    inputs <- data.matrix(x)

    #We need to check the Net object.
    if (class(NET) != "BLNN")
      stop("Make sure your Network is a BLNN object, use BLNN_Build to create the BLNN object.")

    NET <- NET

    ##### Checking if restimation of hyperparameters using evidence procedure is called
    if (evidence) {
      if (is.null(ev.x)) {
        ev.x = inputs
        ev.y = Actual

      }
      ev.out <- .evidence(NET, ev.y, ev.x, itter = 10)
      ngroup <- ev.out[[4]]
      NET$scale.error <- ev.out[[2]]
      NET$scale.weights <-
        list(ev.out[[1]][1:(ngroup - 3)], ev.out[[1]][ngroup - 2], ev.out[[1]][ngroup -
                                                                                 1], ev.out[[1]][ngroup])
    }
    # end of evidence procedure.



    warmup <- floor(iter / 2)
    n.params <- length(.VecNetWts(NET))
    par.names <- as.character(c(1:n.params))

    #Error helper function
    U <- function(NewWts) {
      UpNet    <- .UpNetWts(wts = NewWts, NET)
      UpNetFwd <- .ffwd(UpNet, inputs)
      UpErr    <- .NetErr(Actual, UpNetFwd, inputs)
      UpErr
    }
    #Gradient helper function
    grad.U <- function(NewWts) {
      UpNet <- .UpNetWts(wts = NewWts, NET)
      UpDeriv <- .derivs(Actual, UpNet, inputs)
      UND1 <- c(t(UpDeriv[[1]]))
      UND2 <- c(t(UpDeriv[[2]]))
      VecDeriv <- c(UND1, UND2)
      VecDeriv
    }


    ####################
    ## Argument checking.
    if (is.null(init)) {
      if (chains > 1)
        warning(
          'Using same starting values for each chain -- strongly recommended to use dispersed inits'
        )
      init <- lapply(1:chains, function(i)
        as.numeric(.VecNetWts(NET)))
    } else if (is.function(init)) {
      init <- lapply(1:chains, function(i)
        unlist(init()))
    } else if (length(init) != chains) {
      stop("Length of init does not equal number of chains.")
    } else if (any(unlist(lapply(init, function(x)
      length(unlist(x)) != n.params)))) {
      stop("Initial parameter vector is wrong length")
    }

    if (is.null(seeds)) {
      seeds <- as.integer(runif(chains, 1, 100000))
    } else if (length(seeds != chains))
      stop("Length of seeds does not equal number of chains.")

    algorithm <-
      match.arg(algorithm, choices = c("NUTS", "HMC", "BFGS"))
    if (iter < 10 | !is.numeric(iter))
      stop("iter must be > 10")
    ######################
    ######################



    mcmc.out <- list()
    #Check for parallel
    if (!parallel) {

      if (algorithm == "HMC") {
        mcmc.out <-
          lapply(1:chains, function(i)
            #replace this by our fuction
            .sample_NN_hmc(
              iter = iter,
              fn = U,
              gr = grad.U,
              init = init[[i]],
              warmup = warmup,
              chain = i,
              thin = thin,
              seed = seeds[i],
              control = control, display
            ))
      } else if (algorithm == "NUTS") {
        mcmc.out <- lapply(1:chains, function(i)
          .sample_NN_nuts(
            iter = iter,
            fn = U,
            gr = grad.U,
            init = init[[i]],
            warmup = warmup,
            chain = i,
            thin = thin,
            seed = seeds[i],
            control = control, display
          ))


      } else if (algorithm == "BFGS") {
        bfgs.out <- .BLNN_BFGS(NET, Actual, inputs, iter)
        net.out <- .UpNetWts(NET, bfgs.out$par)
        return(
          list(
            Network = net.out,
            Final_error = bfgs.out$value,
            convergence = bfgs.out$convergence,
            hess = bfgs.out$hessian
          )
        )
        stop("End of training network using BFGS algorithm")

        #Parallel excution
    }}else {
        cat("inside parallel if")
        if (!requireNamespace("snowfall", quietly = TRUE))
          stop("snowfall package not found")
        stopifnot(is.character(path))
        if (file.exists('mcmc_progress.txt'))
          trash <- file.remove('mcmc_progress.txt')
        snowfall::sfInit(parallel = TRUE,
                         cpus = cores,
                         slaveOutfile = 'mcmc_progress.txt')
        ## snowfall::sfLibrary("TMB")
        snowfall::sfExportAll()
        on.exit(snowfall::sfStop())
        message("Starting parallel chains... ")
        ##mcmc.out <- lapply(1:chains, function(i)
        mcmc.out <- snowfall::sfLapply(1:chains, function(i)
          .sample_NN.parallel(
            parallel_number = i,
            path = path,
            iter = iter,
            fn = U,
            gr = grad.U,
            init = init[[i]],
            warmup = warmup,
            chain = i,
            thin = thin,
            seed = seeds[i],
            control = control, display,...

          ))
        message("... Finished parallel chains")
      }

    warmup <- mcmc.out[[1]]$warmup
    ## Clean up returned output
    iters <- unlist(lapply(mcmc.out, function(x) dim(x$samples)[1]))
    if(any(iters!=iter/thin)){
      N <- min(iters)
      warning(paste("Variable chain lengths, truncating to minimum=", N))
    } else {
      N <- iter/thin
    }
    samples <- array(NA, dim=c(N, chains, 1+length(par.names)),
                     dimnames=list(NULL, NULL, c(par.names,'Er__')))

    for(i in 1:chains)
      samples[,i,] <- mcmc.out[[i]]$par[1:N,]
    ## Before transforming, get estimated covariance to be used as metrix
    ## later.
    covar.est <-
      cov(do.call(rbind, lapply(1:chains, function(i)
        mcmc.out[[i]]$par[-(1:warmup), 1:n.params])))
    dimnames(covar.est) <- NULL

    message("... Calculating ESS and Rhat")
    temp <-
      (rstan::monitor(
        samples,
        warmup = warmup,
        probs = .5,
        print = FALSE
      ))
    Rhat <- temp[, 6]
    ess <- temp[, 5]
    if(algorithm=="NUTS"){
      sampler_params <-
      lapply(mcmc.out, function(x) x$sampler_params[1:N,])
    }else {
      sampler_params <- lapply(mcmc.out, function(x) x$sampler_params[1:N,])}


    time.warmup <- unlist(lapply(mcmc.out, function(x) as.numeric(x$time.warmup)))
    time.total <- unlist(lapply(mcmc.out, function(x) as.numeric(x$time.total)))
    cmd <- unlist(lapply(mcmc.out, function(x) x$cmd))

    if(N < warmup) warning("Duration too short to finish warmup period")

    result <- list(
      samples = samples,
      sampler_params = sampler_params,
      time.warmup = time.warmup,
      time.total = time.total,
      algorithm = algorithm,
      warmup = warmup,
      covar.est = covar.est,
      Rhat = Rhat,
      ess = ess
    )
    if (algorithm == "NUTS"){
      result$max_treedepth <- mcmc.out[[1]]$max_treedepth
      #f.result <- with(result,
                       #shinystan::as.shinystan(samples, warmup = warmup,
                       #max_treedepth = max_treedepth, sampler_params = sampler_params,
                       #algorithm = "NUTS", model_name = model))

    }
      #f.result <- with(result,
                        #shinystan::as.shinystan(samples, warmup = warmup,
                        #sampler_params = sampler_params, algorithm = "HMC",
                        #model_name = model))}


    return(invisible(result))

  }

##### SAMPLE HMC ###########

.sample_NN_hmc <-
  function(iter = iter,
           fn = U,
           gr = grad.U,
           init = init[[i]],
           warmup = warmup,
           chain = i,
           thin = thin,
           seed = seeds[i],
           control = control, display=0, path = getwd(),...) {
    #Initialize arguments
    #

    if (!is.null(seed))
      set.seed(seed)
    control <- .update_control(control)

    adapt_delta = control$adapt_delta
    M = control$momentum.mass
    useDA = control$useDA

    gamma = control$gamma
    t0 = control$t0
    kappa = control$kappa
    M1 = control$metric

    eps = control$stepsize
    Lambda = control$Lambda

    init <- as.vector(unlist(init))
    npar <- length(init)

    # adjust the momentum mass vector
    M_inv <- 1 / M
    accepted <- divergence <- Er <- rep(NA, iter)
    ## This holds the rotated but untransformed variables ("y" space)
    theta.out <- matrix(NA, nrow = iter, ncol = npar)

    ## If using covariance matrix and Cholesky decomposition, redefine
    ## these functions to include this transformation. The algorithm will
    ## work in the transformed space
    if(!is.null(M1)){
      ## Using a mass matrix means redefining what fn and gr do and
      ## backtransforming the initial value.
      rotation <- .rotate_space(
        fn = fn,
        gr = gr,
        M = M1,
        y.cur = init
      )
      fn2 <- rotation$fn2
      gr2 <- rotation$gr2
      current.q <- rotation$x.cur
      chd <- rotation$chd
      fn2 <- function(theta) fn(chd %*% init)
      gr2 <- function(theta) as.vector( t( gr(chd %*% init) ) %*% chd )
      chd <- t(chol(M1))               # lower triangular Cholesky decomp.
      chd.inv <- solve(chd)               # inverse
      current.q <- chd.inv %*% init
    } else {
      fn2 <- fn; gr2 <- gr
      current.q <- init
    }



    sampler_params <-
      matrix(
        numeric(0),
        nrow = iter,
        ncol = 4,
        # holds DA info by iteration
        dimnames = list(
          NULL,
          c("accept_stat__",
            "stepsize__", "int_time__", "energy__")
        )
      )

    epsvec <- Hbar <- epsbar <- rep(NA, length = warmup + 1)

    eps <- epsvec[1] <- epsbar[1] <-
      .find.epsilon(theta = init, fn, gr, eps, verbose = FALSE, M_inv)
    if(display>= 0) cat("Initial Step size :", eps , "\n")
    mu <- log(4.005 * eps)
    Hbar[1] <- 0
    gamma <- gamma
    t0 <- t0
    kappa <- kappa



    time.start <- Sys.time()
    message('')
    message(paste('Starting HMC', time.start))
    for (m in 1:iter) {
      L <- max(1, round(Lambda / eps))
      if(display>=2) cat("Number of Leaps :", L , "\n")
      theta.out[m,] <- current.q
      Er[m] <- if (m == 1)
        fn(current.q)
      else
        Er[m - 1]
      p.cur <- p.new <- rnorm(length(current.q), 0, sqrt(M))
      q.new <- current.q
      current.K = sum(M_inv * p.cur ^ 2) / 2
      current.H = fn(current.q) + current.K

      if (useDA & m > warmup)
        eps = eps * runif(1, 0.8, 1.1)
      ## Make a half step for first iteration
      p.new <- p.new + eps * gr(q.new) / 2
      for (i in 1:L) {
        #theta.leapfrog[i,] <- current.q
        #r.leapfrog[i,] <- r.new
        p.new <- p.new + eps * gr(q.new) / 2
        q.new <- q.new + eps * p.new
        p.new <- p.new + eps * gr(q.new) / 2

        ## If divergence, stop trajectory earlier to save computation
        if (any(!is.finite(p.new)) | any(!is.finite(q.new)))
          break
      }
      ## half step for momentum at the end
      p.new <- p.new + eps * gr(q.new) / 2

      proposed.U = fn(q.new)
      proposed.K = sum(M_inv * p.new ^ 2) / 2
      proposed.H = proposed.U + proposed.K
      if(display == 3) cat("Iteration :",m, "---", "(O.Eng, N.Eng)", " (",current.H,proposed.H,")", "\n")
      acceptProb = (current.H - proposed.H)
      ## Numerical divergence is registered as a NaN above. In this case we
      ## want to reject the proposal, mark the divergence, and adjust the
      ## step size down if still adapting (see below).
      if (!is.finite(acceptProb)) {
        divergence[m] <- 1
        acceptProb <- -Inf
      } else {
        divergence[m] <- 0
      }
      if (is.finite(acceptProb) & log(runif(1)) < acceptProb) {
        #print("ACCEPT")
        current.q <- q.new
        accepted[m] <- TRUE
        if (display >=1)
          cat("New Error:" , fn(current.q), "\n")
      } else {
        ## otherwise reject it and stay there
        accepted[m] <- FALSE
        #cat("\n", "proposed U ", proposed.U )
      }

      theta.out[m,] <- current.q
      Er[m] <- fn(current.q)
      if (useDA) {
        ## Do the adapting of eps.
        if (m <= warmup) {
          Hbar[m + 1] <-
            (1 - 1 / (m + t0)) * Hbar[m] + (adapt_delta - min(1, exp(acceptProb))) /
            (m + t0)
          logeps <- mu - sqrt(m) * Hbar[m + 1] / gamma
          epsvec[m + 1] <- exp(logeps)
          logepsbar <-
            m ^ (-kappa) * logeps + (1 - m ^ (-kappa)) * log(epsbar[m])
          epsbar[m + 1] <- exp(logepsbar)
          eps <- epsvec[m + 1]
          if(display >=2) cat("\n", "step size during adapt :", eps, "\n")
        } else {
          eps <- epsbar[warmup]
        }
      }


      ## Save adaptation info.
      sampler_params[m,] <-
        c(min(1, exp(acceptProb)), eps, eps * L, fn(current.q))
      if (m == warmup)
        time.warmup <-
        difftime(Sys.time(), time.start, units = 'secs')
      .print.mcmc.progress(m, iter, warmup, chain)
      ## end of MCMC loop
    }
    ## Back transform parameters if metric is used
    if (!is.null(M1)) {
      theta.out <- t(apply(theta.out, 1, function(x)
        chd %*% x))
    }
    theta.out <- cbind(theta.out, Er)
    theta.out <- theta.out[seq(1, nrow(theta.out), by=thin),]
    sampler_params <- sampler_params[seq(1, nrow(sampler_params), by=thin),]
    if (sum(divergence[-(1:warmup)]) > 0)
      message(paste0(
        "There were ",
        sum(divergence[-(1:warmup)]),
        " divergent transitions after warmup"
      ))
    message(paste0(
      "Final acceptance ratio=",
      sprintf("%.2f", mean(accepted[-(1:warmup)])),
      " and target is ",
      adapt_delta
    ))
    if (useDA)
      message(paste0(
        "Final step size=",
        round(epsbar[warmup], 3),
        "; after ",
        warmup,
        " warmup iterations"
      ))
    time.total <- difftime(Sys.time(), time.start, units = 'secs')
    .print.mcmc.timing(time.warmup = time.warmup, time.total = time.total)
    return(
      fit <- list(
        par = theta.out,
        sampler_params = sampler_params,
        time.total = time.total,
        time.warmup = time.warmup,
        warmup = warmup / thin
      )

    )


  }


### SAMPLE NUTS###########################################

.sample_NN_nuts <-
  function(iter,
           fn,
           gr,
           init,
           warmup = floor((iter) / 2),
           chain = 1,
           thin = 1,
           seed = NULL,
           control = NULL, display=0, path=getwd(),
           ...) {
    ## Now contains all required NUTS arguments



    if (!is.null(seed))
      set.seed(seed)
    control <- .update_control(control)

    adapt_delta = control$adapt_delta
    M = control$momentum.mass
    useDA = control$useDA

    gamma = control$gamma
    t0 = control$t0
    kappa = control$kappa

    eps <- control$stepsize
    w1 = control$w1
    w2 = control$w2
    w3 = control$w3

    init <- as.vector(unlist(init))
    npar <- length(init)
    # adjust the momentum mass vector
    M_inv <- 1 / M

    max_td <- control$max_treedepth
    #adapt_delta <- control$adapt_delta


    # For adapt Mass computation
    M1 <- control$metric
    if (is.null(M1))
      M1 <- rep(1, len = npar)
    if (!(is.vector(M1) | is.matrix(M1)))
      stop("Metric must be vector or matrix")

    adapt_mass <- control$adapt_mass
    ## Mass matrix adapatation algorithm arguments. Same as Stan defaults.
    w1 <- control$w1
    w2 <- control$w2
    w3 <- control$w3
    aws <- w2 # adapt window size
    anw <- w1 + w2 # adapt next window
    if (warmup < (w1 + w2 + w3) & adapt_mass) {
      warning("Too few warmup iterations to do mass matrix adaptation.. disabled")
      adapt_mass <- FALSE
    }
    ## Using a mass matrix means redefining what fn and gr do and
    ## backtransforming the initial value.
    rotation <- .rotate_space(
      fn = fn,
      gr = gr,
      M = M1,
      y.cur = init
    )
    fn2 <- rotation$fn2
    gr2 <- rotation$gr2
    theta.cur <- rotation$x.cur
    chd <- rotation$chd

    sampler_params <-
      matrix(
        numeric(0),
        nrow = iter,
        ncol = 6,
        dimnames = list(
          NULL,
          c(
            "accept_stat__",
            "stepsize__",
            "treedepth__",
            "n_leapfrog__",
            "divergent__",
            "energy__"
          )
        )
      )

    ## This holds the rotated but untransformed variables ("y" space)
    theta.out <- matrix(NA, nrow = iter, ncol = npar)
    #cat("\n","I am in line 201 in Sample NN nutts")

    ## how many steps were taken at each iteration, useful for tuning
    j.results <- Er <- rep(NA, len = iter)
    #cat("\n","I am in line 205 in Sample NN nutts")

    #useDA <- is.null(eps)     # whether to use DA algorithm
    if(useDA){

    epsvec <- Hbar <- epsbar <- rep(NA, length = warmup + 1)

    eps <- epsvec[1] <- epsbar[1] <-
      .find.epsilon(theta = init, fn2, gr2, eps, verbose = FALSE, M_inv)
    cat("pick epsilon", eps , "\n")
    mu <- log(15.05 * eps)
    Hbar[1] <- 0
    gamma <- gamma
    t0 <- t0
    kappa <- kappa
    } else {
    ## dummy values to return
     epsvec <- epsbar <- Hbar <- NULL
    }
    #cat("\n","I am in line 219 in Sample NN nutts")



    ## Start of MCMC chain
    time.start <- Sys.time()
    message('')
    message(paste('Starting NUTS at', time.start))
    for (m in 1:iter) {
      ## Initialize this iteration from previous in case divergence at first
      ## treebuilding. If successful trajectory they are overwritten
      theta.minus <- theta.plus <- theta.cur <- init

      theta.out[m,] <- theta.cur
      Er[m] <- if (m == 1)
        fn2(theta.cur)
      else
        Er[m - 1]
      r.cur <- r.plus <- r.minus <-  rnorm(npar, 0, sqrt(M))
      #cat("\n current M", M , "\n current M_inv", M_inv)
      H0 <- .calculate.H(theta = theta.cur,
                         r = r.cur,
                         fn = fn2,
                         M_inv)

      ## Draw a slice variable u in log space
      logu <-
        log(runif(1)) + .calculate.H(theta = theta.cur,
                                     r = r.cur,
                                     fn = fn2,
                                     M_inv)
      j <- 0
      n <- 1
      s <- 1
      divergent <- 0

      ## Track steps and divergences; updated inside .buildtree
      info <- as.environment(list(n.calls = 0, divergent = 0))
      while (s == 1) {
        v <- sample(x = c(1, -1), size = 1)
        if (v == 1) {
          ## move in right direction
          res <-
            .buildtree(
              theta = theta.plus,
              r = r.plus,
              logu = logu,
              v = v,
              j = j,
              eps = eps,
              H0 = H0,
              fn = fn2,
              gr = gr2,
              info = info,
              M_inv = M_inv
            )
          theta.plus <- res$theta.plus
          r.plus <- res$r.plus
        } else {
          ## move in left direction
          res <-
            .buildtree(
              theta = theta.minus,
              r = r.minus,
              logu = logu,
              v = v,
              j = j,
              eps = eps,
              H0 = H0,
              fn = fn2,
              gr = gr2,
              info = info,
              M_inv = M_inv
            )
          theta.minus <- res$theta.minus
          r.minus <- res$r.minus
        }

        ## test whether to accept this state
        if (!is.finite(res$s))
          res$s <- 0
        if (res$s == 1) {
          if (runif(n = 1,
                    min = 0,
                    max = 1) <= res$n / n) {
            theta.cur <- res$theta.prime
            if (display == 3) {
              cat("New Error", fn2(theta.cur), "\n")
            }
            Er[m] <- fn2(theta.cur)
            ## save accepted parameters
            theta.out[m,] <-
              if (is.vector(M1))
                chd * theta.cur
            else
              t(chd %*% theta.cur)

          }
        }
        n <- n + res$n
        s <-
          as.vector(res$s * .test.nuts(theta.plus, theta.minus, r.plus, r.minus))
        if (!is.finite(s))
          s <- 0
        j <- j + 1
        ## Stop doubling if too many or it's diverged enough
        if (j >= max_td) {
          warning("j larger than max_treedepth, skipping to next m")
          break
        }
      }
      j.results[m] <- j - 1

      alpha2 <- res$alpha / res$nalpha
      if (!is.finite(alpha2))
        alpha2 <- 0
      ## Step size adapation with the
      ## Do the adapting of eps.
      if (useDA) {
        if (m <= warmup) {
          ## Adaptation during warmup:
          Hbar[m + 1] <- (1 - 1 / (m + t0)) * Hbar[m] +
            (adapt_delta - alpha2) / (m + t0)
          ## If logalpha not defined, skip this updating step and use
          ## the last one.
          ## if(is.nan(Hbar[m+1])) Hbar[m+1] <- abs(Hbar[m])
          logeps <- mu - sqrt(m) * Hbar[m + 1] / gamma
          epsvec[m + 1] <- exp(logeps)
          logepsbar <-
            m ^ (-kappa) * logeps + (1 - m ^ (-kappa)) * log(epsbar[m])
          epsbar[m + 1] <- exp(logepsbar)
          eps <- epsvec[m + 1]
        } else {
          ## Fix eps for sampling period
          eps <- epsbar[warmup]
        }
      }

      ## ---------------
      ## Do the adaptation of mass matrix. The algorithm is working in X
      ## space but I need to calculate the mass matrix in Y space. So need to
      ## do this coversion in the calcs below.
      if (adapt_mass & .slow_phase(m, warmup, w1, w3)) {
        ## If in slow phase, update running estimate of variances
        ## The Welford running variance calculation, see
        ## https://www.johndcook.com/blog/standard_deviation/
        if (m == w1) {
          ## Initialize algorithm from end of first fast window
          m1 <- theta.out[m,]
          s1 <- rep(0, len = npar)
          k <- 1
        } else if (m == anw) {
          ## If at end of adaptation window, update the mass matrix to the estimated
          ## variances
          M1 <- as.numeric(s1 / (k - 1)) # estimated variance
          ## Update density and gradient functions for new mass matrix
          if (any(!is.finite(M1))) {
            warning("Non-finite estimates in mass matrix adaptation -- reverting to unit")
            M1 <- rep(1, length(M1))
          }
          rotation <-
            .rotate_space(
              fn = fn,
              gr = gr,
              M = M1,
              y.cur = theta.out[m,]
            )
          fn2 <-
            rotation$fn2
          gr2 <- rotation$gr2
          chd <- rotation$chd

          theta.cur <- rotation$x.cur
          ## Reset the running variance calculation
          k <- 1
          m1 <- theta.out[m,]
          s1 <- rep(0, len = npar)
          ## Calculate the next end window. If this overlaps into the final fast
          ## period, it will be stretched to that point (warmup-w3)
          aws <- 2 * aws
          anw <- .compute_next_window(m, anw, warmup, w1, aws, w3)
          ## Find new reasonable eps since it can change dramatically when M
          ## updates
          eps <-
            .find.epsilon(
              theta = theta.cur,
              fn = fn2,
              gr = gr2,
              eps = control$stepsize,
              verbose = FALSE,
              M_inv
            )
          if (!is.null(control$verbose))
            print(
              paste(
                m,
                ": new range(M) is:",
                round(min(M), 5),
                round(max(M), 5),
                ", pars",
                which.min(M),
                which.max(M),
                ", eps=",
                eps
              )
            )
        } else {
          k <- k + 1
          m0 <- m1
          s0 <- s1
          ## Update M and S
          m1 <- m0 + (theta.out[m,] - m0) / k
          s1 <- s0 + (theta.out[m,] - m0) * (theta.out[m,] - m1)
        }
      }
      ## End of mass matrix adaptation
      ##---------------
      sampler_params[m,] <-
        c(alpha2,
          eps,
          j,
          info$n.calls,
          info$divergent,
          fn2(theta.cur))
      if (m == warmup)
        time.warmup <-
        difftime(Sys.time(), time.start, units = 'secs')
      .print.mcmc.progress(m, iter, warmup, chain)
    } ## end of MCMC loop

    ## Process the output for returning
    theta.out <- cbind(theta.out, Er)
    theta.out <- theta.out[seq(1, nrow(theta.out), by = thin),]
    warm <- warmup / thin
    sampler_params <-
      sampler_params[seq(1, nrow(sampler_params), by = thin),]
    ndiv <- sum(sampler_params[-(1:warm), 5])
    if (ndiv > 0)
      message(paste0("There were ", ndiv, " divergent transitions after warmup"))
    msg <-
      paste0("Final acceptance ratio=", sprintf("%.2f", mean(sampler_params[-(1:warm), 1])))
    if (useDA)
      msg <- paste0(msg, ", and target=", adapt_delta)
    message(msg)
    if (useDA)
      message(paste0(
        "Final step size=",
        round(eps, 3),
        "; after ",
        warmup,
        " warmup iterations"
      ))
    time.total <- difftime(Sys.time(), time.start, units = 'secs')
    .print.mcmc.timing(time.warmup = time.warmup, time.total = time.total)
    return(
      list(
        par = theta.out,
        sampler_params = sampler_params,
        time.total = time.total,
        time.warmup = time.warmup,
        warmup = warm,
        max_treedepth = max_td
      )
    )
  }

.rotate_space <- function (fn, gr, M, y.cur)
{
  if (is.matrix(M)) {
    chd <- t(chol(M))
    chd.inv <- solve(chd)
    fn2 <- function(x)
      fn(chd %*% x)
    gr2 <- function(x) {
      as.vector(gr(chd %*% x) %*% chd)
    }
    x.cur <- as.numeric(chd.inv %*% y.cur)
  }
  else if (is.vector(M)) {
    chd <- sqrt(M)
    fn2 <- function(x)
      fn(chd * x)
    gr2 <- function(x)
      as.vector(gr(chd * x)) * chd
    x.cur <- (1 / chd) * y.cur
  }
  else {
    stop("Mass matrix must be vector or matrix")
  }
  return(list(
    gr2 = gr2,
    fn2 = fn2,
    x.cur = x.cur,
    chd = chd
  ))
}


.update_control <- function (control, ...)
{
  default <-
    list(
      #both NUTS and HMC
      adapt_delta = 0.8,
      momentum.mass = 1,
      stepsize = NULL,
      useDA = TRUE,
      gamma = 0.05,
      t0 = 10,
      kappa = 0.75,
      metric = NULL,
      adapt_mass = TRUE,
      w1 = 75,
      w2 = 50,
      w3 = 25,
      #Only NUTS
      max_treedepth = 10,
      #Only HMC
      Lambda = 0.25
    )


  if (is.matrix(control$metric) & !is.null(control$adapt_mass)) {
    if (control$adapt_mass == TRUE) {
      warning("Mass matrix adaptation disabled if metric is a matrix")
    }
    control$adapt_mass <- FALSE
  }

  new <- default
  if (!is.null(control)) {
    for (i in names(control))
      new[[i]] <- control[[i]]
  }
  return(new)
}




## A recursive function that builds a leapfrog trajectory using a balanced
## binary tree.
##
## @references This is from the No-U-Turn sampler with dual averaging
## (algorithm 6) of Hoffman and Gelman (2014).
##
## @details The function repeatedly doubles (in a random direction) until
## either a U-turn occurs or the trajectory becomes unstable. This is the
## 'efficient' version that samples uniformly from the path without storing
## it. Thus the function returns a single proposed value and not the whole
## trajectory.
##
.buildtree <- function(theta,
                       r,
                       logu,
                       v,
                       j,
                       eps,
                       H0,
                       fn,
                       gr,
                       delta.max = 1000,
                       info = environment(),
                       M_inv,
                       ...) {
  if (j == 0) {
    ## ## Useful code for debugging. Returns entire path to global env.
    # if(!exists('theta.trajectory'))
    # theta.trajectory <<- data.frame(step=0, t(theta))
    ## base case, take one step in direction v
    r <- r + (v * eps / 2) * gr(theta)
    theta <- theta + (v * eps) * r
    r <- r + (v * eps / 2) * gr(theta)
    ## verify valid trajectory. Divergences occur if H is NaN, or drifts
    ## too from from true H.
    #cat("\n Inside buildtree")
    H <- .calculate.H(theta = theta, r = r, fn, M_inv)

    n <- logu <= H
    s <- logu < delta.max + H
    if (!is.finite(H) | s == 0) {
      info$divergent <- 1
      s <- 0
    }
    ## Acceptance ratio in log space: (Hnew-Hold)
    logalpha <- H0 - H
    alpha <- min(exp(logalpha), 1)
    info$n.calls <- info$n.calls + 1
    ## theta.trajectory <<-
    ##   rbind(theta.trajectory, data.frame(step=tail(theta.trajectory$step,1),t(theta)))
    return(
      list(
        theta.minus = theta,
        theta.plus = theta,
        theta.prime = theta,
        r.minus = r,
        r.plus = r,
        s = s,
        n = n,
        alpha = alpha,
        nalpha = 1
      )
    )
  } else {
    ## recursion - build left and right subtrees
    xx <-
      .buildtree(
        theta = theta,
        r = r,
        logu = logu,
        v = v,
        j = j - 1,
        eps = eps,
        H0 = H0,
        fn,
        gr,
        info = info,
        M_inv = M_inv
      )
    theta.minus <- xx$theta.minus
    theta.plus <- xx$theta.plus
    theta.prime <- xx$theta.prime
    r.minus <- xx$r.minus
    r.plus <- xx$r.plus
    alpha <- xx$alpha
    nalpha <- xx$nalpha
    s <- xx$s
    if (!is.finite(s))
      s <- 0
    nprime <- xx$n
    ## If it didn't fail, update the above quantities
    if (s == 1) {
      if (v == -1) {
        yy <- .buildtree(
          theta = theta.minus,
          r = r.minus,
          logu = logu,
          v = v,
          j = j - 1,
          eps = eps,
          H0 = H0,
          fn,
          gr,
          info = info,
          M_inv = M_inv
        )
        theta.minus <- yy$theta.minus
        r.minus <- yy$r.minus
      } else {
        yy <- .buildtree(
          theta = theta.plus,
          r = r.plus,
          logu = logu,
          v = v,
          j = j - 1,
          eps = eps,
          H0 = H0,
          fn,
          gr,
          info = info,
          M_inv = M_inv
        )
        theta.plus <- yy$theta.plus
        r.plus <- yy$r.plus
      }
      ### Update elements:
      ## If both slice variables fail you get 0/0.
      nprime <- yy$n + xx$n
      if (!is.finite(nprime)) {
        nprime <- 0
      }
      ## choose whether to keep this theta
      if (nprime > 0)
        if (runif(1) <= yy$n / nprime)
          theta.prime <- yy$theta.prime
      alpha <- xx$alpha + yy$alpha
      nalpha <- xx$nalpha + yy$nalpha
      ## check for valid proposal
      b <- .test.nuts(
        theta.plus = theta.plus,
        theta.minus = theta.minus,
        r.plus = r.plus,

        r.minus = r.minus
      )
      s <- yy$s * b
    }
    return(
      list(
        theta.minus = theta.minus,
        theta.plus = theta.plus,
        theta.prime = theta.prime,
        r.minus = r.minus,
        r.plus = r.plus,
        s = s,
        n = nprime,
        alpha = alpha,
        nalpha = nalpha
      )
    )
  }
}


.calculate.H <-
  function(theta, r, fn, M_inv) {
    fn(theta) + (1 / 2) * sum(M_inv * r ^ 2)
  }



## Test whether a "U-turn" has occured in a branch of the binary tree
## created by \ref\code{.buildtree} function. Returns TRUE if no U-turn,
## FALSE if one occurred
.test.nuts <- function(theta.plus, theta.minus, r.plus, r.minus) {
  theta.temp <- theta.plus - theta.minus
  res <- (crossprod(theta.temp, r.minus) >= 0) *
    (crossprod(theta.temp, r.plus) >= 0)
  return(res)
}

.find.epsilon <-
  function(theta,
           fn,
           gr,
           eps = 1,
           verbose = TRUE,
           M_inv) {
    r <- rnorm(n = length(theta),
               mean = 0,
               sd = sqrt(1/M_inv))
    .calculate.H1 <-
      function(theta, r, fn, M_inv) {
        fn(theta) - (1 / 2) * sum(M_inv * r ^ 2)
      }

    ## Do one leapfrog step
    #cat("I am here in pick eps,")
    r.new <- r + (eps / 2) * gr(theta)
    #cat("I am here after first gr,")
    theta.new <- theta + eps * r.new
    r.new <- r.new + (eps / 2) * gr(theta.new)
    H1 <- .calculate.H1(theta = theta,
                        r = r,
                        fn = fn,
                        M_inv)
    H2 <- .calculate.H1(theta = theta.new,
                        r = r.new,
                        fn = fn,
                        M_inv)
    a <- 2 * (exp(H2) / exp(H1) > .5) - 1
    ## If jumped into bad region, a can be NaN so setup algorithm to keep
    ## halving eps instead of throwing error
    #cat("finished H,")
    if (!is.finite(a))
      a <- -1
    k <- 1
    ## Similarly, keep going if there are infinite values
    while (!is.finite(H1) |
           !is.finite(H2) | a * H2 - a * H1 > -a * log(2)) {
      eps <- (2 ^ a) * eps
      ## Do one leapfrog step
      r.new <- r + (eps / 2) * gr(theta)
      theta.new <- theta + eps * r.new
      r.new <- r.new + (eps / 2) * gr(theta.new)
      H2 <- .calculate.H1(theta = theta.new,
                          r = r.new,
                          fn = fn,
                          M_inv)
      k <- k + 1
      if (k > 50) {
        stop(
          "More than 50 iterations to find reasonable eps. Model is likely misspecified or some other issue."
        )

      }
    }
    if (verbose)
      message(paste("Reasonable epsilon=", eps, "found after", k, "steps"))
    return(invisible(eps))
  }

.print.mcmc.progress <- function (iteration, iter, warmup, chain)
{
  i <- iteration
  refresh <- max(10, floor(iter / 10))
  if (i == 1 | i == iter | i %% refresh == 0) {
    i.width <- formatC(i, width = nchar(iter))
    out <- paste0(
      "Chain ",
      chain,
      ", Iteration: ",
      i.width,
      "/",
      iter,
      " [",
      formatC(floor(100 * (i / iter)), width = 3),
      "%]",
      ifelse(i <= warmup, " (Warmup)", " (Sampling)")
    )
    message(out)
  }
}

.print.mcmc.timing <- function (time.warmup, time.total)
{
  x <- " Elapsed Time: "
  message(paste0(x, sprintf("%.1f", time.warmup), " seconds (Warmup)"))
  message(paste0(
    x,
    sprintf("%.1f", time.total - time.warmup),
    " seconds (Sampling)"
  ))
  message(paste0(x, sprintf("%.1f", time.total), " seconds (Total)"))
}

.slow_phase <- function (i, warmup, w1, w3)
{
  x1 <- i >= w1
  x2 <- i <= (warmup - w3)
  x3 <- i < warmup
  return(x1 & x2 & x3)
}

.compute_next_window <- function (i, anw, warmup, w1, aws, w3)
{
  anw <- i + aws
  if (anw == (warmup - w3))
    return(anw)
  nwb <- anw + 2 * aws
  if (nwb >= warmup - w3) {
    anw <- warmup - w3
  }
  return(anw)
}###
