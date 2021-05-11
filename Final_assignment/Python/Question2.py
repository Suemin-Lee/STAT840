import matplotlib.pyplot as plt
import numpy as np


# Question(a): Simple MC methods

def func(x):
    y = np.sqrt(x)/np.sqrt(x+1) *np.exp(-x)
    return y 


zscore95 = 1.96


print('================ (a) Simple Monte Carlo ==================')
N_size = [100,1000,10000]
y_simple_MC =[]
for i, n_sample in enumerate(N_size):
    u =np.random.uniform(1,3,n_sample)
    y_simple_MC.append([2*func(u)])
    mean_simple_mc = np.mean(y_simple_MC[i])
    variance_np = np.var(y_simple_MC[i])
    SE_simple = np.sqrt(variance_np)/np.sqrt(n_sample)

    lower_b = mean_simple_mc-zscore95*SE_simple
    upper_b = mean_simple_mc+zscore95*SE_simple
    CI95 = [lower_b , upper_b]
    print('')
    print('**** Number of sample = ', n_sample,'  ****')
    print('theta = ', mean_simple_mc, ' Standar Error = ',SE_simple )
    print('95% Confidence intervals = ', CI95)


print()
print('================ (b) Importance Sampling ==================')
plt.figure(figsize=(12,4))

def func_2(x):
    y = np.sqrt(x)/np.sqrt(x+1) *np.exp(-x)/(1/2*(x-1))
    return y 

N_size = [1000,10000,100000]
for i, n_sample in enumerate(N_size):
    y_simple_MC =[]
    # for _ in range(n_sample):
    u = np.random.uniform(0,1,n_sample)
    x = (np.sqrt(4*u)+1)
        # y_simple_MC.append([func_2(x)])
    y_simple_MC = func_2(x)
    mean_simple_mc = np.mean(y_simple_MC)
    variance_np = np.var(y_simple_MC)
    SE_simple = np.sqrt(variance_np)/np.sqrt(n_sample)

    lower_b = mean_simple_mc-zscore95*SE_simple
    upper_b = mean_simple_mc+zscore95*SE_simple
    CI95 = [lower_b , upper_b]
    print('')
    print('**** Number of sample = ', n_sample,'  ****')
    print('theta = ', mean_simple_mc, ' Standar Error = ',SE_simple )
    print('95% Confidence intervals = ', CI95)
    plt.subplot(1,3,i+1)
    plt.hist(y_simple_MC,bins='auto',range=[ 1,4])


plt.show()


print()
print('================ (c) Control Variate ==================')

N_size = [100,1000,10000]
for n_sample in N_size:
    U = np.random.uniform(1,3,size = n_sample)
    U2 = np.random.uniform(1,3,size = n_sample)
    delta = func(U)*2
    delta_star = -2/5*(1/3*U2-1)

    theta_MC = np.mean(delta)
    theta_MC_star = np.mean(delta_star)
    var_MC = np.var(delta)
    var_MC_star = np.var(delta_star)

    covar_matrix = np.cov([delta,delta_star])
    covariance = covar_matrix[0][1]
    alpha = -covariance/var_MC_star

    theta_CV = theta_MC +alpha*(theta_MC_star-0)
    variance_CV = var_MC + alpha**2*var_MC_star + 2*alpha*covariance

    SE_cv = np.sqrt(variance_CV/n_sample)

    lower_b = theta_CV - zscore95*SE_cv
    upper_b = theta_CV + zscore95*SE_cv
    CI95 = [lower_b , upper_b]
    print('')
    print('**** Number of sample = ', n_sample,'  ****')
    print('theta = ', theta_CV, ' Standar Error  = ',SE_cv )
    print('95% Confidence intervals = ', CI95)

print()
print('================ (d) Antitethic Sampling ==================')

N_size = [100,1000,10000]
for n_sample in N_size:
    U =np.random.uniform(0,1,n_sample)

    x1 = 1+ 2*U
    x2 = 3- 2*U
    delta1 = func(x1)*2
    delta2 = func(x2)*2
    AS_theta = (np.mean(delta1)+np.mean(delta2))/2

    cov_as = np.cov((delta1,delta2),bias=True)[0][1]
    AS_var = 1/4*(np.var(delta1)+np.var(delta1)) +1/2 *cov_as

    AS_SE= np.sqrt(AS_var/n_sample)

    lower_b = AS_theta -zscore95*AS_SE
    upper_b = AS_theta +zscore95*AS_SE
    CI95 = [lower_b , upper_b]
    print('')
    print('**** Number of sample = ', n_sample,'  ****')
    print('theta = ', AS_theta, ' Standar Error  = ', AS_SE )
    print('95% Confidence intervals = ', CI95)






# '================ (E) Comparisons =================='

def func(x):
    y = np.sqrt(x)/np.sqrt(x+1) *np.exp(-x)
    return y 

x_range = np.arange(0,5,0.01)
x_range_1=np.arange(0,1,0.1)
plt.figure(figsize=(15,4))
plt.subplot(1,4,1)
plt.plot(x_range,2*func(x_range),label='$f(x)$')
plt.plot(x_range,np.ones_like(x_range)/2,label='g(x) density function',color='red')
plt.plot(np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.plot(3*np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.xlabel('x variables')
plt.ylabel('function(x)')
plt.title('Simple MC')
plt.legend()
plt.tight_layout()


def func_g(x):
    y = (-1/2)*(1-x)
    return y

x_range = np.arange(0,5,0.01)
x_range_1=np.arange(0,1,0.1)
plt.subplot(1,4,2)
plt.plot(x_range,2*func(x_range),label='$f(x)$')
plt.plot(np.arange(1,3,0.1),func_g(np.arange(1,3,0.1)),label='g(x) density function',color='red')
plt.plot(np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.plot(3*np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.xlabel('x variables')
plt.ylabel('function(x)')
plt.legend()
plt.title('Imporant sampling')
plt.tight_layout()


def func_g(x):
    y = (-1/3)*(1/4*x-1)
    return y

x_range = np.arange(0,5,0.01)
x_range_1=np.arange(0,.5,0.1)
plt.subplot(1,4,3)
plt.plot(x_range,func(x_range),label='$f(x)$')
plt.plot(np.arange(0,4,0.1),func_g(np.arange(0,4,0.1)),label='g(x) proposed density',color='red')
plt.plot(np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.plot(3*np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.xlabel('x variables')
plt.ylabel('function(x)')
plt.title('Control Variate')
plt.legend()
plt.tight_layout()



def func(x):
    y = np.sqrt(x)/np.sqrt(x+1) *np.exp(-x)
    return y 

x_range = np.arange(0,5,0.01)
x_range_1=np.arange(0,1,0.1)
plt.subplot(1,4,4)
plt.plot(x_range,2*func(x_range),label='$f(x)$')
plt.plot(x_range,np.ones_like(x_range)/2,label='g(x) density function',color='red')
plt.plot(np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.plot(3*np.ones_like(x_range_1),x_range_1, linestyle=':',color='black')
plt.xlabel('x variables')
plt.ylabel('function(x)')
plt.legend()
plt.title('Antithetic')
plt.tight_layout()
plt.show()



