import numpy as np 
import matplotlib.pyplot as plt


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










# def f(x):
#     y= np.log(1+x**2)*x**4*np.exp(-3*x)
#     return y



# def integr_area(x,dx):
#     A = (f(x)+f(x+dx))/2*dx
#     return A

# dx=0.01
# range_a = np.arange(1,5-dx,dx)

# save_a =[]

# for x in range_a:
#     save_a.append(integr_area(x,dx))

# print(np.sum(save_a))
# # plt.plot(range_a,save_a)
# plt.plot(range_a,f(range_a))
# plt.show()

# print(((3**5)/factorial(4)))


# y=[]
# for _ in range(10000):
#     u = np.random.uniform(0,1)

#     y.append(np.sqrt(4*u)+1)
# plt.hist(y,bins='auto')
# plt.show()


