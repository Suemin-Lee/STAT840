import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm



def cau_func(x):
    y = 1/(np.pi* (1+x**2))
    return y

x_dis =[1] 
N_itr=1000

for i in range(N_itr):
    x_current = x_dis[0] 
    j = np.random.normal(x_current,1)
    r = cau_func(j)/cau_func(x_current)
    if r>= 1: 
        x_dis.append(j)
    elif np.random.uniform(0,1)<r:
        x_dis.append(j)
    else: x_dis.append(x_current)

burn = 100
converging_x_dis = x_dis[burn:]     
plt.hist(converging_x_dis,bins ='auto')
plt.xlabel('x_data')
plt.ylabel('counts')
plt.title('Histogram for using MCMC with N=%i'%N_itr)


median_x = np.median(x_dis)
SE = np.sqrt(np.var(x_dis))/np.sqrt(len(x_dis))

zscore95 = 1.96

lower_b = median_x-zscore95*SE
upper_b = median_x+zscore95*SE
CI95 = [lower_b , upper_b]

print('================ (b) Median and 95 IC ==================')
print('Mean = ', median_x, ' Standar Error = ',SE )
print('95% Confidence intervals = ', CI95)

plt.show()
