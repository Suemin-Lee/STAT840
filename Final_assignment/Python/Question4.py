import numpy as np 
import matplotlib.pyplot as plt


# plt.scatter(x, y, c=t)

num_samples =10000


# Initializer x0, y0
samples = {'x': [0.2], 'y':[1]}
# Gibbs sampling from different distribution
for i in range(num_samples): 
    curr_x = samples['x'][i]
    curr_y = samples['y'][i]
    # Update varibles
    new_x = np.random.gamma(3, 1/(curr_y**2+4))
    param = 1/(1+new_x)
    new_y = np.random.normal(param, np.sqrt(param))
    samples['x'].append(new_x)
    samples['y'].append(new_y)


# Data for two-dimensional scattered points
plt.figure(figsize=(5,4))
t = np.arange(num_samples+1)
plt.scatter(samples['x'], samples['y'], c=t, cmap='plasma')
plt.xlabel('x data')
plt.ylabel('y data')
plt.colorbar()

# Histogram Plot
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.hist(samples['x'],bins='auto')
plt.title('X variables')
plt.xlabel('x')
plt.ylabel('counts')
plt.subplot(1,2,2)
plt.hist(samples['y'],bins='auto')
plt.title('Y variables')
plt.xlabel('y')
plt.ylabel('counts')

# Part D compute the esimator
func =[]
for i in range(num_samples+1): 
    x_val = samples['x'][i]
    y_val = samples['y'][i]
    func.append(x_val**2 * y_val**3 *np.exp(-x_val**2))

estimator = np.mean(func)
variance_fun = np.var(func)
print('Estimator= ', estimator, '    Variance = ',variance_fun)

plt.figure(figsize = (6,5))
x_range = np.arange(0,len(func))
plt.scatter(x_range,func)
plt.title('$E_\pi[X^2 Y^3$ exp$(-X^2)$] =%3.6f ] Scatter plot'%estimator )
plt.ylabel('$[X^2 Y^3$ exp$(-x^2)$] ')

plt.show()
