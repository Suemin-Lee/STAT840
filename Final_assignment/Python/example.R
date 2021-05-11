model  {
  for (i in 1:8) {
    
    y[i] ~ dbin(pi[i],n[i])                    
    x[i] <- (w[i]-mu)/sigma
    logist[i] <- exp(x[i])/(1+exp(x[i]))
    pi[i] <- pow(logist[i],m1)}
  
  m1~ dgamma(1,1)
  mu ~ dnorm(0,0.001)
  tau ~ dgamma(0.001,0.001)
  
  sigma <-1 /sqrt(tau)}


Data

list(y=c(6,13,18,28,52,53,61,60),
     n=c(59,60,62,56,63,59,62,60),
     w=c(1.6907,1.7242,1.7552,1.7842,1.8113,1.8369,1.861,1.8839))

Inits for 3 parallel chains

list(m1=0.5,mu=1.8,tau=1)
list(m1=1,mu=2.0,tau=1000)
list(m1=2,mu=1.0,tau=0.5)


2) Informative priors (from Carlin and Louis)

model  {
  for (i in 1:8) {
    
    y[i] ~ dbin(pi[i],n[i])                       
    x[i] <- (w[i]-mu)/sigma
    logist[i] <- exp(x[i])/(1+exp(x[i]))                        
    pi[i] <- pow(logist[i],m1)}
  
  m1~ dgamma(.25,.25)
  mu ~ dnorm(2,10)
  tau ~ dgamma(2.000004,0.001)
  sigma<-1 /sqrt(tau)}


Data

list(y=c(6,13,18,28,52,53,61,60),
     n=c(59,60,62,56,63,59,62,60),
     w=c(1.6907,1.7242,1.7552,1.7842,1.8113,1.8369,1.861,1.8839))

Inits

list(m1=0.5,mu=1,tau=2)
list(m1=1,mu=2.0,tau=1000)


node	 mean	 sd	 MC error	2.5%	median	97.5%	start	sample
m1	0.3828	0.133	0.007278	0.2019	0.358	0.7226	4001	10000
mu	1.811	0.01122	6.354E-4	1.785	1.811	1.83	4001	10000
sigma	0.01888	0.003523	1.552E-4	0.01299	0.01855	0.02667	4001	10000

autocor for mu
0.95 (lag1), Exponential decay 

crosscor for (mu,sigma), (mu,m1), (m1,sigma)
0.75-0.9  -0.1-0.25  -0.25-5


3) Fixing mu at the empirical value 

model  {
  for (i in 1:8) {
    
    y[i] ~ dbin(pi[i],n[i])                    
    x[i] <- (w[i]-mu)/sigma                       
    logist[i] <- exp(x[i])/(1+exp(x[i]))                        
    pi[i] <- pow(logist[i],m1)}
  
  m1 ~ dgamma(1,1)
  tau ~ dgamma(0.001,0.001)
  sigma <-1 /sqrt(tau)}


Data

list(mu=1.8,
     y=c(6,13,18,28,52,53,61,60),
     n=c(59,60,62,56,63,59,62,60),
     w=c(1.6907,1.7242,1.7552,1.7842,1.8113,1.8369,1.861,1.8839))

Inits

list(m1=0.5,tau=1)

