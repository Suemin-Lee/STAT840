# Question 1 

n_size <- 25
t_samples <-rt(n_size, 3)
p1<- quantile(t_samples,  probs = c(0.25, 0.75))
print(p1<- quantile(t_samples,  probs = c(0.25, 0.75)))



p1<- quantile(t_samples,  probs = 0.25)
p2<- quantile(t_samples,  probs = 0.75)
p1
p2
F <- (p2-p1)/1.34

F

plug_in_fun <- function(x_sample){
  p1<- quantile(t_samples,  probs = c(0.25, 0.75))
  theta <- (p1$0.25%[1]
            [1]-p1[2])/1.34
  return(theta)
}


plug_in_fun(t_samples)



plug_in_fun <- function(x){
  quantiles <- quantile( x, c(.25, .75 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}
plug_in_fun( t_samples )







