import matplotlib.pyplot as plt
import numpy as np


# Question(a): Simple MC methods exponential

def func(x):
    y = np.sqrt(x)/np.sqrt(x+1) 
    return y 

zscore95 = 1.96


print('================ (a) Simple Monte Carlo ==================')
N_size = [100,1000,10000]
y_simple_MC =[]
# for i, n_sample in enumerate(N_size):
N_size =10000
for i in range(N_size):
    u =np.random.exponential(1)
    if u>1 and u<3:
        y_simple_MC.append(func(u))
mean_simple_mc = np.mean(y_simple_MC)
variance_np = np.var(y_simple_MC)
SE_simple = np.sqrt(variance_np)/np.sqrt(N_size)

lower_b = mean_simple_mc-zscore95*SE_simple
upper_b = mean_simple_mc+zscore95*SE_simple
CI95 = [lower_b , upper_b]
print('')
print('**** Number of sample = ', N_size,'  ****')
print('theta = ', mean_simple_mc, ' Standar Error = ',SE_simple )
print('95% Confidence intervals = ', CI95)

print()
print('================ (b) Importance Sampling ==================')


# N_size = [100,1000,10000]
# y_simple_MC =[]
# for i, n_sample in enumerate(N_size):
#     u =np.random.uniform(1,3,n_sample)
#     y_simple_MC.append([func(u)])
#     mean_simple_mc = np.mean(y_simple_MC[i])
#     variance_np = np.var(y_simple_MC[i])
#     SE_simple = np.sqrt(variance_np)/np.sqrt(n_sample)

#     lower_b = mean_simple_mc-zscore95*SE_simple
#     upper_b = mean_simple_mc+zscore95*SE_simple
#     CI95 = [lower_b , upper_b]
#     print('')
#     print('**** Number of sample = ', n_sample,'  ****')
#     print('theta = ', mean_simple_mc, ' Standar Error = ',SE_simple )
#     print('95% Confidence intervals = ', CI95)




# print()
# print('================ (c) Control Variate ==================')

# N_size = [100,1000,10000]
# for n_sample in N_size:
#     U = np.random.uniform(1,3,size = n_sample)
#     U2 = np.random.uniform(1,3,size = n_sample)
#     delta = func(U)*2
#     delta_star = -2/5*(1/3*U2-1)

#     theta_MC = np.mean(delta)
#     theta_MC_star = np.mean(delta_star)
#     var_MC = np.var(delta)
#     var_MC_star = np.var(delta_star)

#     covar_matrix = np.cov([delta,delta_star])
#     covariance = covar_matrix[0][1]
#     alpha = -covariance/var_MC_star

#     theta_CV = theta_MC +alpha*(theta_MC_star-0)
#     variance_CV = var_MC + alpha**2*var_MC_star + 2*alpha*covariance

#     SE_cv = np.sqrt(variance_CV/n_sample)

#     lower_b = theta_CV - zscore95*SE_cv
#     upper_b = theta_CV + zscore95*SE_cv
#     CI95 = [lower_b , upper_b]
#     print('')
#     print('**** Number of sample = ', n_sample,'  ****')
#     print('theta = ', theta_CV, ' Standar Error  = ',SE_cv )
#     print('95% Confidence intervals = ', CI95)

# # print()
# # print('================ (d) Antitethic Sampling ==================')

# # N_size = [100,1000,10000]
# # for n_sample in N_size:
# #     U =np.random.uniform(0,1,n_sample)

# #     x1 = 1+ 2*U
# #     x2 = 3- 2*U
# #     delta1 = func(x1)*2
# #     delta2 = func(x2)*2
# #     AS_theta = (np.mean(delta1)+np.mean(delta2))/2

# #     cov_as = np.cov((delta1,delta2),bias=True)[0][1]
# #     AS_var = 1/4*(np.var(delta1)+np.var(delta1)) +1/2 *cov_as

# #     AS_SE= np.sqrt(AS_var/n_sample)

# #     lower_b = AS_theta -zscore95*AS_SE
# #     upper_b = AS_theta +zscore95*AS_SE
# #     CI95 = [lower_b , upper_b]
# #     print('')
# #     print('**** Number of sample = ', n_sample,'  ****')
# #     print('theta = ', AS_theta, ' Standar Error  = ', AS_SE )
# #     print('95% Confidence intervals = ', CI95)
