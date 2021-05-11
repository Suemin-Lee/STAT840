#Question 7 (a)-(e)

least_square<- function(ab){
  alpha <- ab[1]
  beta  <- ab[2]
  x = c(9.1, 4.41, 18.99, 13.73, 9.1, 30.89, 19.17, 8.64, 
        8.67, 15.62, 14.68 , 11.09 , 11.53, 2.28 , 3.64, 
        5.5 , 0.73, 12.39, 25.7 , 6.31 , 12.43 , 4.81, 
        9.28, 4.82 , 3.85 , 6.88, 12.48, 11.66, 8.06, 5.97) 
  
  y = c(4.49, 4.12, 4.84, 4.93, 4.24, 4.9, 4.81, 4.57, 
        4.37, 4.61, 4.29, 4.55, 4.35, 3.56, 3.76, 4.18, 
        2.25, 4.46, 4.63, 4.27, 4.61, 4.11, 4.74, 3.98, 
        3.96, 4.42, 4.78,4.55, 4.35, 4.25)
  
  n<- length(x)
  regression <-rep(0,n)
  for (i in 1:n){
    regression[i] <- (y[i] - (alpha *x[i])/(1+beta*x[i]) )^2
  }
  least_square<- sum(regression)
  return(least_square)
}

least_square_n<- function(ab){
  alpha <- ab[1]
  beta  <- ab[2]
  x = c(9.1, 4.41, 18.99, 13.73, 9.1, 30.89, 19.17, 8.64, 
        8.67, 15.62, 14.68 , 11.09 , 11.53, 2.28 , 3.64, 
        5.5 , 0.73, 12.39, 25.7 , 6.31 , 12.43 , 4.81, 
        9.28, 4.82 , 3.85 , 6.88, 12.48, 11.66, 8.06, 5.97) 
  
  y = c(4.49, 4.12, 4.84, 4.93, 4.24, 4.9, 4.81, 4.57, 
        4.37, 4.61, 4.29, 4.55, 4.35, 3.56, 3.76, 4.18, 
        2.25, 4.46, 4.63, 4.27, 4.61, 4.11, 4.74, 3.98, 
        3.96, 4.42, 4.78,4.55, 4.35, 4.25)
  
  n<- length(x)
  regression <-rep(0,n)
  for (i in 1:n){
    regression[i] <- (y[i] - (alpha *x[i])/(1+beta*x[i]) )^2
  }
  least_square<- -sum(regression)
  return(least_square)
}

#(a) Newton_Raphson hessian calculated in python
#This is for verification check
n_iteration=30
theta_update <- list()          
theta_update[[1]] <- c(3, 0.5)

for (k in 1:n_iteration){
  theta_current<- theta_update[[k]]
  grad_like <- grad(func=least_square, x=theta_current)
  hess  <-  hessian(func=least_square, x=theta_current)
  inverse_hess <-solve(hess)
  theta_update[[k+1]] <- (theta_current- 
                            (inverse_hess%*% grad_like))
}
optimaized_par<- t(theta_update[[n_iteration+1]])
optimaized_par
least_square(optimaized_par)


initial_value <- c(3, 0.5)

#(b) "Nelder-Mead"
out_optim_Nelder <- optim( par = initial_value, 
                           fn =least_square, 
                           method = "Nelder-Mead")
out_optim_Nelder$par
out_optim_Nelder$value
out_optim_Nelder$counts
out_optim_Nelder$convergence


#(c) Quasi-Newton (BFGS)
out_optim_BFGS <- optim( par = initial_value, 
                         fn =least_square, 
                         method = "BFGS")
out_optim_BFGS$par
out_optim_BFGS$value
out_optim_BFGS$counts
out_optim_BFGS$convergence
out_optim_BFGS$message

#(d) Simulated Anealing algorithm (SANN)
out_optim_SANN <- optim( par = initial_value, 
                         fn =least_square, 
                         method = "SANN")

out_optim_SANN$par
out_optim_SANN$value
out_optim_SANN$counts
out_optim_SANN$convergence
out_optim_SANN$message

#(e) Genetic Algorithm 
library(GA)

GA <- ga(type = "real-valued", 
         fitness =  least_square_n,
         lower = c(2, 0.7), upper = c(8, 2.0), 
         popSize = 30, maxiter = 1000, run = 100)
summary(GA)
least_square(GA@solution)
plot(GA)





