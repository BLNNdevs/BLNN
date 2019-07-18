#' @title Build a Feed-Forward Neural Network Structure
#' @name BLNN_Build
#' @description Allows the user to generate a neural network structure for the purposes of training and predicting.
#' @param ncov An integer, the number of input units.
#' @param nout An integer, the number of output units.
#' @param hlayer_size An integer, the number of nodes making up the hidden layer. Default is 1.
#' @param actF The choice of activation function. See 'Details'.
#' @param costF The choice of cost function. See 'Details'.
#' @param outF The choice of output function. See 'Details'.
#' @param bias The choice for the bias term of each layer. Default is 1 for each layer.
#' @param hp.Err Value of the scale hyperparameter for the network errors. Defaults to FALSE for no bayesian.
#' @param hp.W1 Value of the scale hyperparameter for the first layer weights. Defaults to FALSE for no bayesian.
#' @param hp.B1 Value of the scale hyperparameter for the first layer bias. Defaults to FALSE for no bayesian.
#' @param hp.W2 Value of the scale hyperparameter for the second layer weights. Defaults to FALSE for no bayesian.
#' @param hp.B2 Value of the scale hyperparameter for the second layer bias. Defaults to FALSE for no bayesian.
#' @param decay_term Control term for the initial valus of the weights. Standard deviation of initial weights is 1/decay_term. Default is 1.
#'
#' @details BLNN_Build provides users with different activation, cost, and output finctions
#' which can be chosen based on the model type. Activation functions are applied at the hidden
#' layer in order to assist in computation where the output function restricts the range of values to
#' fit the given problem. We recomend using tanh in the activation function in most cases. MSE can be
#' used when taking the error of linear outputs but cross entropy is suggested otherwise.
#'
#' @return The network object, defined as a list containing
#' \describe{
#' \item{ncov}{Number of covariates.}
#'
#' \item{nout}{Number of outputs.}
#'
#' \item{hidden_size}{Size of the hidden layer.}
#'
#' \item{actF}{Activation Function.}
#'
#' \item{costF}{Cost Function.}
#'
#' \item{outF}{Output Function.}
#'
#' \item{bias}{Bias terms.}
#'
#' \item{scale.error}{Error hyperparameter.}
#'
#' \item{scale.weights}{A list containing weight and bias hyperparameters.}
#'
#' \item{weights}{A list containing each layer's weight matrix.}
#'
#' \item{Amat}{A placeholder to contain weights1 multiplied by inputs.}
#'
#' \item{postAmat}{A placeholder to contain actF evaluated at Amat.}
#'
#' \item{Zmat}{A placeholder to contain weights2 multiplied by postAmat.}
#'
#' \item{trainout}{A placeholder to contain trained output values.}
#' }
#' @export




BLNN_Build<-function(ncov,
                      nout,
                      hlayer_size= 1,
                      actF = c("linear", "sigmoid", "tanh"),
                      costF=c("MSE","crossEntropy"),
                      outF = c("linear", "sigmoid", "softmax"),
                      bias=c(1,1),
                      hp.Err=FALSE,
                      hp.W1=FALSE,
                      hp.W2=FALSE,
                      hp.B1=FALSE,
                      hp.B2=FALSE,
                      decay_term=1){



  nn<-NULL #start with an entirely blank object

  #check that an integer number is provided for number of covariates
  if((length(ncov)>1L & is.null(ncol(ncov))==FALSE & is.null(nrow(ncov))==FALSE) | is.numeric(ncov) == FALSE) stop("Number of covariates must be an integer number")


  nn$ncov<-as.integer(ncov) #set the number of covs to be an integer type instead of the default double type.

  #check that a integer value is provided
  if((length(nout)>1L & is.null(ncol(nout))==FALSE & is.null(nrow(nout))==FALSE) | is.numeric(nout) == FALSE) stop("Number of output variables must be a number")

  nn$nout<-as.integer(nout) #set the number of outputs to be an integer type.

  #Check that the layer size is appropriate
  if(length(hlayer_size)>1 & is.numeric(hlayer_size) == TRUE) stop("Hidden layer size must be integer number")

  nn$hidden_size<-hlayer_size #set the size argument to be the layer sizes, a single value


  #check the activation function (FOR HIDDEN LAYERS UNITS ONLY) (TAKE a AND OUTPUT z)
  if(match(actF,c("linear", "sigmoid", "tanh", "softmax"), nomatch = FALSE)==FALSE) stop("Unavailable activation function")

  nn$actF<-actF

  #check the output function (TAKE z AND OUTPUT y)
  if(match(outF,c("linear", "sigmoid", "softmax"), nomatch = FALSE)==FALSE) stop("Unavailable output function")

  nn$outF<-outF


  #check the error function (FOR NOW MEAN SQUARE ERROR and CROSS ENTROPY, others to be added)
  if(match(costF,c("MSE","crossEntropy"), nomatch = FALSE)==FALSE) stop("Unavailable cost function") #check if an available cost function is supplied

  if(costF == "MSE") nn$costF <- "MSE"

  if(costF=="crossEntropy") nn$costF<- "crossEntropy"

  #Check that the bias terms are numeric values
  if(length(bias)!=2 & is.numeric(bias)==FALSE) stop("Bias terms must be a vector of length two with numeric entries")
  #Save the bias
  nn$bias<-bias

  #Check that the hyper parameter values are non negative or not null
  if(any(c(hp.Err,hp.W1, hp.B1, hp.W2, hp.B2)<=0) & any(c(hp.Err,hp.W1, hp.B1, hp.W2, hp.B2)!=FALSE) &
     is.numeric(c(hp.Err,hp.W1, hp.B1, hp.W2, hp.B2))==TRUE) stop("Hyperparameter values must be non negative real numbers and all specified if any are used")

  #Check that the hyperparameters are of the right length
  if(length(c(hp.Err, hp.B1, hp.W2, hp.B2))!=4) stop("All hyperparameter values other than HP.W1 must be scalars")
  if(length(hp.W1)!=ncov & length(hp.W1)!=1) stop("Hyperparameters for the first layer must be length one or number of covariates")

  #Place the error scale into the network. Will place NULL by default
  nn$scale.error<-hp.Err

  #Fill a list with the elements of the scale weights
  nn$scale.weights<-list(hp.W1, hp.B1, hp.W2, hp.B2)


  ###########################################################################
  ###########################################################################Flip this to after the general method
  ###########################################################################
  #go to set up weights

      nn$weights<-list() #blank for bayesian methods to compute in the training methods

      #if we are not using a hidden layer
      if (hlayer_size==0){

        #set weights if there is no hidden layer
        swts<-rnorm(((ncov+1)*nout), mean=0, sd=1/decay_term )/sqrt(ncov+1)

        #make the single weight matrix
        nn$weights<-matrix(swts, nrow = ncov+1, ncol=nout)
      } else {

      for(lrs in 1:(length(hlayer_size)+1)){ #loop over the number of hidden layers to build a wt matrix for each layer

         #Check that the matrix multiplication can be done by checking the row numbers
         if(lrs!=1 & lrs!=(length(hlayer_size)+1)){

         #set weights to be a vector of the number of connections that need to be made with decay term
         swts<-rnorm( ((hlayer_size[lrs-1]+1)*hlayer_size[lrs]), mean=0, sd=1/decay_term )


         #Take the vector of the random weights and fill the matrix for that given layer. Col's are for connections
         nn$weights[[lrs]]<-matrix(swts, nrow = hlayer_size[lrs-1]+1, ncol=hlayer_size[lrs])
         }

           else if(lrs==1){

           #set weights to be a vector of the number of connections that need to be made with decay term
           swts<-rnorm( ((ncov+1)*hlayer_size[lrs]), mean=0, sd=1/decay_term ) /sqrt(ncov+1)

           #if it is the first matrix of weights the rows should be the covariates plus bias row
           nn$weights[[lrs]]<-matrix(swts, nrow = ncov+1, ncol=hlayer_size[lrs])
           }
               else {
                 swts<-rnorm( ((hlayer_size[lrs-1]+1)*nout), mean=0, sd=1/decay_term )/sqrt(hlayer_size+1)

                 #if it is the first matrix of weights the rows should be number of hidden nodes plus bias
                 nn$weights[[lrs]]<-matrix(swts, nrow = hlayer_size[lrs-1]+1, ncol=nout)

               }

      }


    }


  #set the blank list for pre activated matrices
  nn$Amat<-list()

  #set the blank list of activated a's
  nn$postAmat<-list()

  #set the blank list of Z values
  nn$Zmat<-list()

  #set the blank list of training outputs
  nn$trainout<-list()

  class(nn)<-"BLNN"


  nn #output the network object
}

