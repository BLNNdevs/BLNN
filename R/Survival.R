#' Well-known data set from Appendix D of Fleming and Harrington (1991) taken from Terry Therneau's website. We have copied the data set and here is SAS code to read in the data and run a standard Cox regression. Note that in the SAS code we eliminate cases past 312 and recode status=1 (liver transplant) to be status=0 (alive, i.e., censored). There are only 276 complete cases when running a Cox regression with all covariates. From Terry Therneau's website, the variables in the data set are
#'  id       = case number
#'  futime   = number of days between registration and the earlier of death,
#' transplantion, or study analysis time in July, 1986
#' status   = 0=alive, 1=liver transplant, 2=dead
#' drug     = 1= D-penicillamine, 2=placebo
#' age      = age in days
#' sex      = 0=male, 1=female
#' ascites  = presence of ascites: 0=no 1=yes
#' hepato   = presence of hepatomegaly 0=no 1=yes
#' spiders  = presence of spiders 0=no 1=yes
#' edema    = presence of edema 0=no edema and no diuretic therapy for edema;
#' .5 = edema present without diuretics, or edema resolved by diuretics;
#' 1 = edema despite diuretic therapy
#' bili     = serum bilirubin in mg/dl
#' chol     = serum cholesterol in mg/dl
#' albumin  = albumin in gm/dl
#' copper   = urine copper in ug/day
#' alk_phos = alkaline phosphatase in U/liter
#' sgot     = SGOT in U/ml
#' trig     = triglicerides in mg/dl
#' platelet = platelets per cubic ml/1000
#' protime  = prothrombin time in seconds
#' stage    = histologic stage of disease
#'
#' @docType data
#'
#' @usage data(PBC_surv)
#'
#' @format An object of class \code{"data.frame"}.
#'
#' @keywords datasets
#'
#'
#' @source \href{https://www4.stat.ncsu.edu/~boos/var.select/pbc.html}
