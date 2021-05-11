import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm


plt.figure(figsize=(16,5))
n_size_range=[100,1000,10000];
print('=============== Question 10 (b) =====================')

for i, n_size in enumerate(n_size_range):
    theta=[]; theta_y=[];
    for _ in range(n_size):
        y = np.random.normal(0,1)

        x_y = np.random.normal(y ,1+y**2)
        theta_y.append(y)
        theta.append(x_y)
    Expectation = np.mean(theta)
    SE = np.sqrt(np.var(theta))/np.sqrt(n_size)
    print()
    print('N samples =',n_size)
    print('marginal estimator',Expectation,'SE = ', SE)

    plt.subplot(1,3 ,i+1)
    plt.hist(theta,bins='auto',range=[ -15,15],label='X random variables')
    plt.hist(theta_y,bins='auto',range=[ -15,15],label='Y random variables')
    plt.title('N samples = %i'%n_size)
    plt.legend()

plt.tight_layout()
plt.show()