
#####################################################################################################################
# Function to update the net and the weights
#####################################################################################################################
.UpNetWts<-function(wts,NET){


  if(NET[["hidden_size"]]>0){
    NewWts<-wts
    #set split point for the weight vector
    bkpt<-(NET[["ncov"]]+1)*NET[["hidden_size"]]

    #split the single weight vector into two based on the breakpoint
    w1<-matrix(NewWts[1:bkpt], ncol=NET[["hidden_size"]], byrow = TRUE)
    w2<-matrix(NewWts[(bkpt+1):length(NewWts)], ncol=NET[["nout"]], byrow = TRUE)

    #update the network object
    NET[["weights"]][[1]]<-w1
    NET[["weights"]][[2]]<-w2

    #Output the network
    NET

  }else{
    NewWts<-wts

    w1<-matrix(NewWts, ncol=NET[["nout"]], byrow = TRUE)

    NET[["weights"]]<-w1

    NET
  }
}



.VecNetWts<-function(NET){

  if(NET[["hidden_size"]]>0){
    #Vector for first weights
    w1<-c(t(NET[["weights"]][[1]]))
    w2<-c(t(NET[["weights"]][[2]]))

    #Full vector of all net weights
    VecW<-c(w1,w2)

    #output vector of weights
    VecW

  }else{
    w1<-c(t(NET[["weights"]]))
    w1
  }
}

#####################################################################################################################
# Function to compute the network error
#####################################################################################################################
.NetErr<-function(Actual, NET, inputs,sep=FALSE, ...){

  #Take the network and feed forward with the most recent set of weights
  fwded<-.ffwd(NET, inputs)

  #Extract the predicted values
  y<-fwded[["trainout"]]

  #Switch over the error functions
  if(NET[["costF"]]=="crossEntropy"){

    #For Cross Entropy
    Err<-(-sum(Actual*log(y)+(1-Actual)*log(1-y)))
  } else{

    #For MSE
    Err<-mean((Actual-y)^2)
  }

  if(sep==TRUE){

    BRS<-.ErrBayes(NET, Err, sep=TRUE)

    return(list(Total=Err+BRS, Bayes=BRS, Data=Err))

  }

  else{
    #Return the final net error
    FinalErr<-.ErrBayes(NET, Err)

    return(FinalErr)}
}





#####################################################################################################################
# Function to forward the network
#####################################################################################################################
.ffwd<-function(NET,data, ...){

  #define an empty function to hold our activation function based on name in NET

  func<-NULL

  if(NET[["actF"]]=="linear") func<-quote(a) #Assign the formula based on name

  if(NET[["actF"]]=="sigmoid") func<-quote(1/(1+exp(-a))) #Assign the formula based on name

  if(NET[["actF"]]=="tanh") func<-quote(tanh(a)) #Assign the formula based on name

  if(NET[["actF"]]=="softmax") func<-quote(exp(a)/sum(exp(a))) #Assign the formula based on name




  #save the covatiates as a for easy itterating. This will be overwritten to the real a immediatly
  a<-data.matrix(data)

  #if we have no hidden layers
  if(NET[["hidden_size"]][1]==0){
    #do nothing and move on
    NULL

  } else{

    for(i in 1:length(NET[["hidden_size"]])){

      #multiply the imputs by the weights to get the connection matrix
      a<-cbind(a,rep(NET[["bias"]][[i]], nrow(a)))%*%(NET[["weights"]][[i]])

      if(NET[["actF"]]=="tanh") {

        NET[["Amat"]][[i]]<-a

        a <-eval(func)
        #matrix hidden units output
        NET[["postAmat"]][[i]]<-a
      }

      else if(NET[["actF"]]=="sigmoid") {
        #Ensure that log(1-y) is computable

        lrgst <- -log(.Machine$double.eps)

        # Ensure that log(y) is computable

        smlst <- -log(1/.Machine$double.xmin - 1)

        #if the value of a is too big to get the exponential

        a <- pmin(a, lrgst)

        #if the value of a is too small to get the exponential
        a <- pmax(a, smlst)

        #save the value of a before sigmoid
        NET[["Amat"]][[i]]<-a

        #evaluate with the activation function to a
        a<-eval(func)

        #save the activated values
        NET[["postAmat"]][[i]]<-a
      }

      else if(NET[["actF"]]=="softmax"){
        #Ensure that sum(exp(a), 2) does not overflow
        lrgst <- log(.Machine$double.xmin) - log(NET[["nout"]])
        #Ensure that exp(a) > 0
        smlst <- log(.Machine$double.xmin)
        a <- pmin(a, lrgst)
        a <- pmax(a, smlst)

        #save the value of a before softmax
        NET[["Amat"]][[i]]<-a

        #evaluate with the activation function to a
        a<-eval(func)

        #save the activated values
        NET[["postAmat"]][[i]]<-a
      }

    }
  }

  #define an empty function to hold our output function based on name in NET
  func1<-NULL

  if(NET[["outF"]]=="linear") func1<-quote(z) #Assign the formula based on name

  if(NET[["outF"]]=="sigmoid") func1<-quote(1/(1+exp(-z))) #Assign the formula based on name

  if(NET[["outF"]]=="softmax") func1<-quote(exp(z)/rowSums(exp(z))) #Assign the formula based on name


  #if we dont have a hidden layer
  if(NET[["hidden_size"]][1]==0){

    z<-cbind(a,rep(NET[["bias"]], nrow(a)))%*%(NET[["weights"]])


  }else{

    #multiply the inputs by the weights to get the connection matrix
    z<-cbind(a,rep(NET[["bias"]][[length(NET[["weights"]])]], nrow(a)))%*%(NET[["weights"]][[length(NET[["weights"]])]])
  }

  if(NET[["outF"]]=="sigmoid") {
    #Ensure that log(1-y) is computable

    lrgst <- -log(.Machine$double.eps)

    # Ensure that log(y) is computable

    smlst <- -log(1/.Machine$double.xmin - 1)

    #if the value of a is too big to get the exponential

    z <- pmin(z, lrgst)

    #if the value of a is too small to get the exponential
    z <- pmax(z, smlst)

    #save the z matrix
    NET[["Zmat"]]<-z
  }

  if(NET[["outF"]]=="softmax"){
    #Ensure that sum(exp(a), 2) does not overflow
    lrgst <- log(.Machine$double.xmax) - log(NET[["nout"]])
    #Ensure that exp(a) > 0
    smlst <- log(.Machine$double.xmin)
    z <- pmin(z, lrgst)
    z <- pmax(z, smlst)

    #save the z matrix
    NET[["Zmat"]]<-z

  }

  #evaluate with the activation function to a
  y<-eval(func1)

  #save training outs
  NET[["trainout"]]<-y

  NET

  #print(y)
}




#####################################################################################################################
# Function to compute the network gradient
#####################################################################################################################
.derivs<-function(Actual, NET, inputs){

  #Feed forward the net
  NET<-.ffwd(NET, inputs)

  ######### h'(a)#################################
  #set up necessary dervs of activation functions
  if(NET[["actF"]]=="linear") DactF<-quote(a-a+1)

  if(NET[["actF"]]=="sigmoid") DactF<-quote(a)*(1-a)

  if(NET[["actF"]]=="tanh") DactF<-quote(1-a^2)

  if(NET[["actF"]]=="softmax") DactF<-quote(exp(a[i])/sum(exp(a))*D-exp(a[j])/sum(exp(a)))
  #where D needs to be 0 if i \neq j and 1 if i=j
  #########################################################################################


  if(NET[["hidden_size"]]>0){

    a<-NET[["Amat"]][[1]]
    y<-NET[["trainout"]]
    pa<-NET[["postAmat"]][[1]]


    #multiply pointwise to get output errors
    outError<-(y-Actual)
    Gw2<-t(pa)%*%outError
    Gb2<-colSums(outError)



    ############################BACKPROP##############################

    #generate a blank list for rate of change for cost with respect to weights
    RCweights<-list()

    #if there are no hidden units
    if(length(NET[["hidden_size"]])==0) {

      RCweight<-inputs%*%outError

    } else {
      #set blank list to hold each layers errors
      layerErrors<-list()

      #set the final element of the layer errors to be the output errors
      layerErrors[[length(NET[["weights"]])]]<- outError

      #Itterate over the remaining layers
      for (q in (length(NET[["weights"]]))-1:1) {

        #generate the layer errors from the given formula

        layerErrors[[q]]<-(layerErrors[[q+1]])%*%t(NET[["weights"]][[q+1]][1:nrow(NET[["weights"]][[q+1]])-1,])*eval(DactF)
      }

      RCweights[[1]]<-t(inputs)%*%layerErrors[[1]]
      Gb1<-colSums(layerErrors[[1]])
      RCweights[[1]]<-rbind(RCweights[[1]], Gb1)
      RCweights[[2]]<-rbind(Gw2, Gb2)


    }

    #Compute final gradient
    FinalGrad<-.DBayes(NET, RCweights)

    #Return final gradient
    FinalGrad
  }

  else{
    y<-NET[["trainout"]]

    #multiply pointwise to get output errors
    outError<-(y-Actual)
    Gw<-t(inputs)%*%outError
    Gb<-colSums(outError)
    RCweights<-rbind(Gw, Gb)

    #Compute final gradient
    FinalGrad<-.DBayes(NET, RCweights)

    #Return final gradient
    FinalGrad
  }
}









#####################################################################################################################
# Function to compute the bayesian dervitive
#####################################################################################################################

.DBayes<-function(NET, DatErr,...){

  #get data derivitives
  gdata<-DatErr

  #If a error scale is provided multiply it by the weights
  if(NET[["scale.error"]]==TRUE){
    gdata<-lapply(gdata, "*", NET[["scale.error"]])
  }
  #Otherwise the data gradient stays the just the data component
  else {
    NULL
  }

  if(NET[["hidden_size"]]>0){
    #if we are not using bayesian
    if(NET[["scale.weights"]][[1]]==FALSE){
      gprior<-0
    }

    #If the input weights only have one scale
    else if(length(NET[["scale.weights"]][[1]])==1){

      #Multiply the weights by the same scale value for all inputs (plus the first layer bias scale)
      #We repeat the same values for the number of inputs
      w1pri<-NET[["weights"]][[1]]*
        c(rep(NET[["scale.weights"]][[1]],
              nrow(NET[["weights"]][[1]])-1),
          NET[["scale.weights"]][[2]])

      #Multiply the weights by the same scale value for all inputs (plus the second layer bias scale)
      w2pri<-NET[["weights"]][[2]]*
        c(rep(NET[["scale.weights"]][[3]],
              (nrow(NET[["weights"]][[2]])-1)),
          NET[["scale.weights"]][[4]])

      #Return list of prior grad
      gprior<-list(w1pri, w2pri)

    }
    #Othersise, scale for each input will be different and thus
    else{

      #Multiply by corresponding values. We do not use rep as the we have a vector for each input scale and bias
      w1pri<-NET[["weights"]][[1]]*
        c(NET[["scale.weights"]][[1]],
          NET[["scale.weights"]][[2]])

      #Multiply the weights by the same scale value for all inputs (plus the second layer bias scale)
      w2pri<-NET[["weights"]][[2]]*
        c(rep(NET[["scale.weights"]][[3]],
              (nrow(NET[["weights"]][[2]])-1)),
          NET[["scale.weights"]][[4]])

      #Return list of prior grad
      gprior<-list(w1pri, w2pri)
    }

    #Sum the two lists to get full bayesian grad matrices for each set of weights
    BayesDeriv<-mapply("+", gdata, gprior)

    #Return the bayesian grad
    BayesDeriv

  }else{

    if(NET[["scale.weights"]][[1]]==FALSE){
      gprior<-0
    }

    #If the input weights only have one scale
    else if(length(NET[["scale.weights"]][[1]])==1){

      #Multiply the weights by the same scale value for all inputs (plus the first layer bias scale)
      #We repeat the same values for the number of inputs
      w1pri<-NET[["weights"]]*
        c(rep(NET[["scale.weights"]][[1]],
              nrow(NET[["weights"]])-1),
          NET[["scale.weights"]][[2]])

      #Return list of prior grad
      gprior<-list(w1pri)

    }
    #Othersise, scale for each input will be different and thus
    else{

      #Multiply by corresponding values. We do not use rep as the we have a vector for each input scale and bias
      w1pri<-NET[["weights"]][[1]]*
        c(NET[["scale.weights"]][[1]],
          NET[["scale.weights"]][[2]])

      #Return list of prior grad
      gprior<-list(w1pri)
    }

    #Sum the two lists to get full bayesian grad matrices for each set of weights
    BayesDeriv<-mapply("+", gdata, gprior)

    #Return the bayesian grad
    BayesDeriv

  }
}








#####################################################################################################################
# Function to compute the bayesian error
#####################################################################################################################

.ErrBayes <- function(NET, NetErr,sep=FALSE, ...) {
  #get data errors
  edata <- NetErr

  #If an error scale is provided, multiply it by the weights
  if (NET[["scale.error"]] == TRUE) {
    edata <- edata * NET[["scale.error"]]

    #Otherwise do nothing
  } else{
    NULL
  }





  if (NET[["hidden_size"]] > 0) {
    #If we are not using bayesian, have the prior error be zero
    if (NET[["scale.weights"]][[1]][1] == FALSE) {
      eprior <- 0

      #If all covariates have the same distribution of weights
    } else if (length(NET[["scale.weights"]][[1]]) == 1) {
      #Extract each set of weights and quadratic error
      w1 <- .5 * (NET[["weights"]][[1]]) ^ 2
      w2 <- .5 * (NET[["weights"]][[2]]) ^ 2

      #Multiply the transformed weights by the appropriate alpha value
      w1t <- w1 * c(rep(NET[["scale.weights"]][[1]],
                        nrow(NET[["weights"]][[1]]) - 1),
                    NET[["scale.weights"]][[2]])

      w2t <- w2 * c(rep(NET[["scale.weights"]][[3]],
                        nrow(NET[["weights"]][[2]]) - 1),
                    NET[["scale.weights"]][[4]])

      #Take the sum of all weight errors with respect to the bayesian
      eprior <- sum(c(sum(w1t), sum(w2t)))

      #Otherwise if the first layer has different distributions for each covariate
    } else{
      #Extract each set of weights and quadratic error
      w1 <- .5 * (NET[["weights"]][[1]]) ^ 2
      w2 <- .5 * (NET[["weights"]][[2]]) ^ 2

      #Multiply the transformed weights by the appropriate alpha value
      w1t <- w1 * c(NET[["scale.weights"]][[1]],
                    NET[["scale.weights"]][[2]])

      w2t <- w2 * c(rep(NET[["scale.weights"]][[3]],
                        nrow(NET[["weights"]][[2]]) - 1),
                    NET[["scale.weights"]][[4]])

      #Take the sum of all weight errors with respect to the bayesian
      eprior <- sum(c(sum(w1t), sum(w2t)))
    }

    #Finally output the bayesian error term
    return(eprior + edata)

    #In the case we don't have a hidden layer
  } else{
    #If we are not using bayesian, have the prior error be zero
    if (NET[["scale.weights"]][[1]][1] == FALSE) {
      eprior <- 0

      #If all covariates have the same distribution of weights
    } else if (length(NET[["scale.weights"]][[1]]) == 1) {
      #Extract each set of weights and quadratic error
      w1 <- .5 * (NET[["weights"]]) ^ 2

      #Multiply the transformed weights by the appropriate alpha value
      w1t <- w1 * c(rep(NET[["scale.weights"]][[1]],
                        nrow(NET[["weights"]]) - 1),
                    NET[["scale.weights"]][[2]])

      #Take the sum of all weight errors with respect to the bayesian
      eprior <- sum(c(sum(w1t)))

      #Otherwise if the first layer has different distributions for each covariate
    } else{
      #Extract each set of weights and quadratic error
      w1 <- .5 * (NET[["weights"]]) ^ 2

      #Multiply the transformed weights by the appropriate alpha value
      w1t <- w1 * c(NET[["scale.weights"]][[1]],
                    NET[["scale.weights"]][[2]])

      #Take the sum of all weight errors with respect to the bayesian
      eprior <- sum(c(sum(w1t)))
    }
    if(sep==TRUE){
      return(eprior)
    }
    else{#Finally output the bayesian error term
      return(edata + eprior)
    }

  }

}



#####################################################################################################################
# Function for the evidence procedure
#####################################################################################################################
.evidence<-function(net, actual, Data, itter=10){

  #Save the elements in the function environment
  network<-net
  act<-actual
  data<-Data


  #hold my place
  hold_my_place<-.BLNN_BFGS(network, act, data)


  #Grab the data component of the hess
  dhess<-hold_my_place$hessian

  #Grab the weights
  weights<-hold_my_place$par

  network<-.UpNetWts(weights,network)

  #Get the data and bayes errors seperatly
  allerr<-.NetErr(act,network, data, sep=TRUE)

  edata<-allerr[[3]]
  eprior<-allerr[[2]]


  #Make Room
  allerr<-NULL
  hold_my_place<-NULL

  #Get the eigenvectors
  evec<-eigen(dhess)$vectors


  #Build diagonal matrix of eigen values
  evl<-eigen(dhess)$values


  #Get number of parameters
  nparam<-length(evl)

  #Set all negatives to zero
  evl<-evl*(evl>0)

  #Set the values for the computation of log evidence
  eps_evl<-evl+.Machine$double.eps*(evl<=0)

  #Put the eigen values into a diagonal matrix
  evl<-diag(evl, nrow =nparam, ncol=nparam)

  #Define the number of groups of hypers for the weights
  if(length(network$scale.weights)<4) message(print("We suggest using a different training method with the absence of a hidden layer"))
  ngroup<-length(network$scale.weights[[1]])+
    length(network$scale.weights[[3]])+
    length(network$scale.weights[[2]])+
    length(network$scale.weights[[4]])
  gams<-rep(0, ngroup)
  logas<-rep(0, ngroup)

  #reconstruct the hessian using the new corrected eigen values
  dh<-evec%*%evl%*%t(evec)

  #Initialize the counter
  counter<-0


  while(counter<itter){

    #Calculate the inverse of the hessian
    hinv<-solve(dhess)


    #Count the number of weights in each group
    if(length(network$scale.weights[[1]])>1){
      n_in_groups<-c((rep(network$hidden_size, network$ncov)),
                     network$hidden_size,
                     network$hidden_size,
                     network$nout)
    } else {
      n_in_groups<-c(network$ncov*network$hidden_size,
                     network$hidden_size,
                     network$hidden_size,
                     network$nout)
    }


    #Create the vector of all the alpha values
    allalpha<-c(network$scale.weights[[1]],
                network$scale.weights[[2]],
                network$scale.weights[[3]],
                network$scale.weights[[4]])

    if(length(n_in_groups)!=length(allalpha)) message(print("length of first layer hyperparameters does not match number of covariates or is not length 1"))

    #Diag hess sum elements asociated with each group
    HessDiagGroupSum<-vector()
    breaker<-1
    #Calculate the sum of the diag hess elements for each group
    for (g in 1:length(n_in_groups)) {
      HessDiagGroupSum<-sum(diag(hinv)[breaker:breaker+n_in_groups[g]-1])
      breaker<-n_in_groups[g]+breaker
    }

    gams<-n_in_groups-allalpha*HessDiagGroupSum

    allalpha<-Re(gams/(2*eprior))

    logas<-.5*n_in_groups*log(allalpha)

    gamma<-sum(gams)
    logev<-sum(logas)


    betaval<-.5*(network$nout*nrow(data)-gamma)/edata
    logev<-logev+.5*nrow(data)*log(betaval)-.5*nrow(data)*log(2*pi)
    local_beta<-betaval

    #Update the network structure for the next while itteration

    network$scale.error<-betaval



    #Ideally this is how the alphas are structured but may be an issue
    network$scale.weights<-list(allalpha[1:(ngroup-3)], allalpha[ngroup-2], allalpha[ngroup-1], allalpha[ngroup])


    #Update counter
    counter<-counter+1
  }


  #Return the entire network error

  error<-.NetErr(act, network, data)


  splitter<-1

  for (d in 1:length(n_in_groups)) {
    GroupEigens<-sum(eps_evl[splitter:splitter+n_in_groups[d]-1])
    splitter<-n_in_groups[g]+splitter
  }

  for(b in 1:ngroup){
    logev<-logev-error-0.5*sum(log(local_beta*GroupEigens+allalpha[b]))
  }

  return(list(alpha_vals=allalpha, beta_val=betaval, log_evidence=logev, ngroup=ngroup))
}



#####################################################################################################################
# Function for pick epsilon
#####################################################################################################################
.PickEps<-function(q, NET, x, y, n, g){

  q0<-q

  e<-5 #starting epsilon

  p0<-rnorm(n) #starting momentum

  res<-.leapfrog(q1=q0, p1=p0, eps1=e, g1=g, x1=x, y1=y, NET1=NET)

  uold<-U(NET, q0, x, y)-(.5*sum((p0)^2))
  unew<-U(NET, wts=res$q, x, y)-(.5*sum((res$p)^2))
  prob<-exp(unew)/exp(uold)

  a<-ifelse(prob>0.5, 1, -1)

  if (!is.finite(a))
    a <- -1

  while (!is.finite(uold) | !is.finite(unew) | (prob^a)>(-a *log(2))) {

    e<-(2^a)*e
    res<-.leapfrog(q1=q0, p1=p0, eps1=e, g1=g, x1=x, y1=y, NET1=NET)
    unew<-U(NET, res$q, x, y)-(.5*sum((res$p)^2))
    prob<-unew/uold
  }
  return(e)
}


#####################################################################################################################
# Function for leapfrog
#####################################################################################################################
.leapfrog<-function(q1, p1, eps1, g1, x1, y1, NET1){

  p1 = p1 + eps1/2*g1 #update momentum
  q1 = q1 + eps1*p1
  g1 = grad.U(NET=NET1, Actual=y1, inputs=x1, wts=q1) #get grad
  p1 = p1 + eps1/2*g1

  res<-list()
  res$p<-p1
  res$q<-q1
  res$g<-g1

  return(res)
}


#####################################################################################################################
# Function for BFGS training
#####################################################################################################################
.BLNN_BFGS<-function(NET, actual, inputs, iter = 200){

  NETWORK<-NET
  ACTUAL<-actual
  INPUTS<-inputs
  init.wts<-.VecNetWts(NET)


  fn1<-function(init.wts){

    network<-.UpNetWts(NETWORK, wts = init.wts)
    #Take the network and feed forward with the most recent set of weights
    fwded<-.ffwd(network, INPUTS)
    #Extract the predicted values
    y<-fwded[["trainout"]]

    #Switch over the error functions
    if(network[["costF"]]=="crossEntropy"){

      #For Cross Entropy
      Err<-(-sum(ACTUAL*log(y)+(1-ACTUAL)*log(1-y)))
    } else{

      #For MSE
      Err<-mean((ACTUAL-y)^2)
    }
    #Return the final net error
    FinalErr<-.ErrBayes(network, Err)

    FinalErr
  }





  gr1<-function(init.wts){

    net<-.UpNetWts(NETWORK, wts = init.wts)

    #Feed forward the net
    network<-.ffwd(net, INPUTS)

    ######### h'(a)#################################
    #set up necessary dervs of activation functions
    if(network[["actF"]]=="linear") DactF<-quote(a-a+1)

    if(network[["actF"]]=="sigmoid") DactF<-quote(a)*(1-a)

    if(network[["actF"]]=="tanh") DactF<-quote(1-a^2)

    if(network[["actF"]]=="softmax") DactF<-quote(exp(a[i])/sum(exp(a))*D-exp(a[j])/sum(exp(a)))
    #where D needs to be 0 if i \neq j and 1 if i=j
    #########################################################################################


    if(network[["hidden_size"]]>0){

      a<-network[["Amat"]][[1]]
      y<-network[["trainout"]]
      pa<-network[["postAmat"]][[1]]


      #multiply pointwise to get output errors
      outError<-(y-ACTUAL)
      Gw2<-t(pa)%*%outError
      Gb2<-colSums(outError)



      ############################BACKPROP##############################

      #generate a blank list for rate of change for cost with respect to weights
      RCweights<-list()

      #if there are no hidden units
      if(length(network[["hidden_size"]])==0) {

        RCweight<-INPUTS%*%outError

      } else {
        #set blank list to hold each layers errors
        layerErrors<-list()

        #set the final element of the layer errors to be the output errors
        layerErrors[[length(network[["weights"]])]]<- outError

        #Itterate over the remaining layers
        for (q in (length(network[["weights"]]))-1:1) {

          #generate the layer errors from the given formula

          layerErrors[[q]]<-(layerErrors[[q+1]])%*%t(network[["weights"]][[q+1]][1:nrow(network[["weights"]][[q+1]])-1,])*eval(DactF)
        }

        RCweights[[1]]<-t(INPUTS)%*%layerErrors[[1]]
        Gb1<-colSums(layerErrors[[1]])
        RCweights[[1]]<-rbind(RCweights[[1]], Gb1)
        RCweights[[2]]<-rbind(Gw2, Gb2)


      }

      #Compute final gradient
      FinalGrad<-.DBayes(network, RCweights)

      UND1<-c(t(FinalGrad[[1]]))
      UND2<-c(t(FinalGrad[[2]]))
      #Full vector of all net derivs
      VecDeriv<-c(UND1,UND2)
      #Output the new vector of derivitives
      return(VecDeriv)
    }

    else{
      y<-network[["trainout"]]

      #multiply pointwise to get output errors
      outError<-(y-ACTUAL)
      Gw<-t(INPUTS)%*%outError
      Gb<-colSums(outError)
      RCweights<-rbind(Gw, Gb)

      #Compute final gradient
      FinalGrad<-.DBayes(NET, RCweights)

      #Return final gradient
      FinalGrad
    }
  }

  #nparam<-

  output<-optim(init.wts,fn1, gr1, method = "BFGS", hessian = TRUE, control = list(maxit = iter))


  return(output)
}




.sample_NN.parallel<- function(parallel_number, path,iter ,
                               fn ,
                               gr ,
                               init ,
                               warmup,
                               chain ,
                               thin ,
                               seed ,
                               control , display...){

  olddir <- getwd()
  on.exit((setwd(olddir)))
  newdir <- paste0(file.path(getwd,path), "_chain_" , parallel_number)
  if (dir.exists(newdir)) {
    unlink(newdir, TRUE)
    if (dir.exists(newdir))
      stop(paste("Could not remove folder:", newdir))
  }
  dir.create(newdir)
  trash <- file.copy(from = list.files(path, full.names = TRUE),
                     to = newdir)
  if (algorithm == "NUTS")
    fit <- .sample_NN_nuts(iter = iter,
                           fn = fn,
                           gr = gr,
                           init = init,
                           warmup = warmup,
                           chain = parallel_number,
                           thin = thin,
                           seed = seed,
                           control = control, display, path = newdir,...)
  if (algorithm == "HMC")
    fit <- .sample_NN_hmc(iter = iter,
                          fn = U,
                          gr = grad.U,
                          init = init[[i]],
                          warmup = warmup,
                          chain = parallel_number,
                          thin = thin,
                          seed = seeds[i],
                          control = control, display, path = newdir,...)
  unlink(newdir, TRUE)
  return(fit)
}
