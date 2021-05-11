#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tabulate import tabulate
import seaborn as sns


x_data =np.loadtxt('waiting.txt',unpack=True)

plt.figure(figsize=(5,4))
plt.hist(x_data,bins =13)
plt.xlabel('time(sec)')
plt.ylabel('counts')
plt.title('waiting time observation')
# plt.show()


def w_i(x,pi,mu1,mu2,sigma1,sigma2):
        f1 = norm.pdf(x, mu1, sigma1) 
        f2 = norm.pdf(x, mu2, sigma2) 
        ans = pi*f1/((pi*f1)+((1-pi)*f2))
        return ans

def pi_k1_val(x_data,pi,mu1,mu2,sigma1,sigma2):
        n = len(x_data)
        W=[]
        for x in x_data:
                W.append(w_i(x,pi,mu1,mu2,sigma1,sigma2))
        ans = sum(W)/n
        return  ans  

def mu1_k1_val(x_data,pi,mu1,mu2,sigma1,sigma2):
        n = len(x_data)
        W_n=[]; W_d=[]
        for x in x_data:
                w_val = w_i(x,pi,mu1,mu2,sigma1,sigma2)
                W_n.append(w_val*x)
                W_d.append(w_val)
        return sum(W_n)/(sum(W_d))

def mu2_k1_val(x_data,pi,mu1,mu2,sigma1,sigma2):
        n = len(x_data)
        W_n=[]; W_d=[]
        for x in x_data:
                w_val = w_i(x,pi,mu1,mu2,sigma1,sigma2)
                W_n.append((1-w_val)*x)
                W_d.append((1-w_val))
        return sum(W_n)/(sum(W_d))

def sigma1_k1_val(x_data,pi,mu1,mu2,sigma1,sigma2,new_mu1):
        n = len(x_data)
        W_n=[]; W_d=[]
        for x in x_data:
                w_val = w_i(x,pi,mu1,mu2,sigma1,sigma2)
                W_n.append(w_val*(x-new_mu1)**2)
                W_d.append(w_val)
        return np.sqrt(sum(W_n)/(sum(W_d)))

def sigma2_k1_val(x_data,pi,mu1,mu2,sigma1,sigma2,new_mu2):
        n = len(x_data)
        W_n=[]; W_d=[]
        for x in x_data:
                w_val = w_i(x,pi,mu1,mu2,sigma1,sigma2)
                W_n.append((1-w_val)*(x-new_mu2)**2)
                W_d.append((1-w_val))
        return np.sqrt(sum(W_n)/(sum(W_d)))

def log_likelihood(x_data,pi,mu1,mu2,sigma1,sigma2):
        n = len(x_data)
        f_obs =[]
        for x in x_data:
                f1 =  norm.pdf(x, mu1, sigma1) 
                f2 =  norm.pdf(x, mu2, sigma2) 
                f_obs.append((pi*f1)+((1-pi)*f2))
        ans = np.prod(f_obs)
        return ans

#Initial choice
pi_range =[0.6, 0.4, 0.6, 0.7]
mu1_range =[45,50,50,60]
mu2_range = [70,75,80,85]
sigma1_range = [10,5,8,10]

# pi_range =[0.6]
# mu1_range =[45]
# mu2_range = [70]
# sigma1_range = [10]


print('=========Question 4 (c)============')
print('')

for j in range(len(pi_range)):

    # theta = {'pi': [0.6], 'mu1':[45], 'mu2':[70],'sigma1':[10],'sigma2':[10]}
    theta = {'pi': [pi_range[j]], 
            'mu1':[mu1_range[j]], 'mu2':[mu2_range[j]],
            'sigma1':[sigma1_range[j]],'sigma2':[sigma1_range[j]]}
    sub_theta = {'pi': [], 'mu1':[], 'mu2':[],'sigma1':[],'sigma2':[]}
    num_samples = 25

    # eps =10**(-4) #Epsilon value choose
    for i in range(num_samples+1): 
            curr_pi  = theta['pi'][i]
            curr_mu1 = theta['mu1'][i]
            curr_mu2 = theta['mu2'][i]
            curr_sigma1 = theta['sigma1'][i]
            curr_sigma2 = theta['sigma2'][i]

            new_pi = pi_k1_val(x_data, curr_pi, curr_mu1, curr_mu2, curr_sigma1, curr_sigma2)
            new_mu1 = mu1_k1_val(x_data, curr_pi, curr_mu1, curr_mu2, curr_sigma1, curr_sigma2)
            new_mu2 = mu2_k1_val(x_data, curr_pi, curr_mu1, curr_mu2, curr_sigma1, curr_sigma2)
            new_sigma1 = sigma1_k1_val(x_data, curr_pi, curr_mu1, curr_mu2, curr_sigma1, curr_sigma2,new_mu1)
            new_sigma2 = sigma2_k1_val(x_data, curr_pi, curr_mu1, curr_mu2, curr_sigma1, curr_sigma2, new_mu2)

            Q_kth = log_likelihood(x_data, curr_pi, curr_mu1, curr_mu2, curr_sigma1, curr_sigma2)
            Q_k_1_th = log_likelihood(x_data, curr_pi, new_mu1, new_mu2 ,new_sigma1,new_sigma2)

            theta['pi'].append(new_pi)
            theta['mu1'].append(new_mu1)
            theta['mu2'].append(new_mu2)
            theta['sigma1'].append(new_sigma1)
            theta['sigma2'].append(new_sigma2)
            iteration = [1, 2, 3, 5, 10, 15, 20, 25]
            if i in iteration: 
                sub_theta['pi'].append(new_pi)
                sub_theta['mu1'].append(new_mu1)
                sub_theta['mu2'].append(new_mu2)
                sub_theta['sigma1'].append(new_sigma1)
                sub_theta['sigma2'].append(new_sigma2)
            # if abs(Q_k_1_th-Q_kth)<=eps:
            #         break
    print('=========Initial value choice========')
    print('[pi, mu1, mu2, sigma1 , sigma2] =  [', theta['pi'][0],',',theta['mu1'][0],',',theta['mu2'][0], ',',theta['sigma1'][0],',',theta['sigma2'][0], ']')

    print(tabulate(sub_theta, headers='keys', showindex = iteration,tablefmt ='fancy_grid'))

    print('========= Covergent values 25th========')
    print('[pi, mu1, mu2, sigma1 , sigma2] =  [', theta['pi'][-1],',',theta['mu1'][-1],',',theta['mu2'][-1], ',',theta['sigma1'][-1],',',theta['sigma2'][-1], ']')
    print('--------------------------------------------------------------------------------------------------------')
    print('')


# plot figure 

# def f_function(pi, mu1, mu2, sigma1 , sigma2 ):
plt.figure(figsize =(5,4))
def log_likelihood(x_data,pi,mu1,mu2,sigma1,sigma2):
        n = len(x_data)
        f_obs =[]
        for x in x_data:
            f1 =  norm.pdf(x, mu1, sigma1) 
            f2 =  norm.pdf(x, mu2, sigma2) 
            f_obs.append((pi*f1)+((1-pi)*f2))
        ans = f_obs
        return ans

y_data = log_likelihood(x_data,theta['pi'][-1]
                        ,theta['mu1'][-1],theta['mu2'][-1],
                        theta['sigma1'][-1],theta['sigma2'][-1])

new_x, new_y = zip(*sorted(zip(x_data, y_data)))


fig = plt.figure(figsize=(5,4))
sns.distplot(x_data,kde=False,norm_hist=True)
plt.scatter(x_data,y_data,color='skyblue')
plt.plot(new_x, new_y,color='blue')
plt.xlabel('time(sec)')
plt.ylabel('counts')
plt.title('waiting time observation')
fig.legend(labels=[r'f(x|$\hat{\theta}$) (line)'
                   ,r'f(x|$\hat{\theta}$) (scatter)'
                   ,' histogram of $x_{obs}$'], bbox_to_anchor=(0.53, 0.9))
plt.tight_layout()
plt.show()







