import matplotlib.pyplot as plt
import numpy as np



x_data = [186, 181, 176, 149, 184, 190, 158, 139, 175, 148, 
        152, 111, 141, 153, 190, 157, 131, 149, 135, 132,]

zscore90 = 1.645 
n_size = len(x_data)

# Question (a)
theta_hat = np.mean(x_data)
variance_hat = np.var(x_data)
SE_normal =np.sqrt(variance_hat)/n_size
lower_b = theta_hat-zscore90*SE_normal
upper_b = theta_hat+zscore90*SE_normal
CI90 = [lower_b , upper_b]

print('================ Original data analysis==================')
print('===== (a) Mean and variance with 90CI  === ')
print('Mean = ', theta_hat, 'variance = ',variance_hat )
print('90% Confidence intervals = ', CI90)
print('')

# Question (b) Jack Knife
Jack_theta =[]; 
for index in range(n_size):
    new_t= x_data[:index] + x_data[index+1 :]
    theat_hat_star = np.mean(new_t)
    Jack_theta.append(theat_hat_star)
jack_SE = np.sqrt(((n_size-1)/n_size)*np.sum((Jack_theta-theta_hat)**2))
lower_b = theta_hat - zscore90*jack_SE; 
upper_b = theta_hat + zscore90*jack_SE;
CI90_jack = [lower_b , upper_b]

print('================ Jackknife Method ==================')
print('===== (b) Normal inveral with Standar Error  === ')
print('Average confidence_length SE = ', CI90_jack)
print('')



# Question Bootstrap (c) and (d)
B=10000
Boot_theta=[]
for _ in range(B):
    new_dis_sample=[];
    for _ in range(n_size):
        indx = np.random.uniform(0,len(x_data))
        indx = int(indx)
        new_dis_sample.append(x_data[indx])
    Boot_theta.append(np.mean(new_dis_sample))
# Normal interval standard error methods
Boot_SE = np.sqrt(np.var(Boot_theta))
lower_b = theta_hat - zscore90*Boot_SE; 
upper_b = theta_hat + zscore90*Boot_SE;
boot_SE_CI = [lower_b,upper_b]

# 90 percentile intervals 
boot_q095 = np.quantile(Boot_theta, 0.95);
boot_q005 = np.quantile(Boot_theta, 0.05);
boot_percent_CI = [boot_q005,boot_q095] 


print('================ Bootstrap Method ==================')
print('===== (d) Normal inveral with SE with B=10000  === ')
print('Average confidence_length SE = ', boot_SE_CI)
print('')
print('===== (c) Percentile invterals with B=10000  === ')
print('Average confidence_length SE = ', boot_percent_CI)
print('')
