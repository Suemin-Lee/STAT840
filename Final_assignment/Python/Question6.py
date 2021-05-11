import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x_data = [186, 181, 176, 149, 184, 190, 158, 139, 175, 148, 
        152, 111, 141, 153, 190, 157, 131, 149, 135, 132]
x_size = len(x_data)
x_med = np.median(x_data)
x_mean = np.mean(x_data)
x_std = np.std(x_data)
numm_H = 165
# Z - score
z_scores = (x_med-numm_H)/(x_std)/np.sqrt(x_size)
p_values = norm.sf(abs(z_scores)) 

print('======= Question6 (a)========')

print('X data mean = ', x_mean)
print('X data median = ', x_med)
print('X data standard deviation = ', x_std)
print()
print('Z-score = ', z_scores)
print('p-value = ', p_values)

# Question 7 (b)
print('======= Question6 (b) questions before revisions ========')
for i in range(5):
    B=2**10
    n_size = len(x_data)
    null_H = 165
    Perm_theta=[]
    Perm_pi_vector=[]
    for _ in range(B):
        new_dis_sample=[]; pi_vec =[];
        for _ in range(n_size):
            indx = np.random.uniform(0,len(x_data))
            indx = int(indx)
            new_dis_sample.append(x_data[indx])
            if x_data[indx]>null_H:
                pi_vec.append(1)
            else: pi_vec.append(-1)
        Perm_theta.append(np.median(new_dis_sample))
        Perm_pi_vector.append(np.median(pi_vec))


    larger_elem = [elem for elem in Perm_theta if elem > x_med]
    count_ele = len(larger_elem)

    prob = (count_ele)/B
    print('')
    print('*****',(i+1),'th Trial***** ')
    print('P-value = ',prob)

print()

# Question 6 (b)
print('======= Question6 (b) after revision from the question========')

B=2**10
n_size = len(x_data)
null_H = 165

T_obs_data = [x-null_H for x in x_data]
T_obs = np.mean(T_obs_data)

for i in range(5):
    Perm_theta=[]
    Perm_pi_vector=[]
    for _ in range(B):
        new_dis_sample=[]; pi_vec =[];
        for _ in range(n_size):
            indx = np.random.uniform(0,len(x_data))
            indx = int(indx)
            new_dis_sample.append(x_data[indx])
        T_b = [x_i- null_H for x_i in new_dis_sample]
        Perm_theta.append(np.mean(T_b))
    larger_elem = [elem for elem in Perm_theta if elem > T_obs]
    count_ele = len(larger_elem)

    prob = (count_ele)/B
    print('')
    print('*****',(i+1),'th Trial***** ')
    print('P-value = ',prob)
