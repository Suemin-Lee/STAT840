import matplotlib.pyplot as plt

x_data = [ 9.1, 4.41, 18.99, 13.73, 9.1, 30.89, 19.17, 8.64, 8.67, 15.62,
        14.68 , 11.09 , 11.53, 2.28 , 3.64, 5.5 , 0.73, 12.39, 25.7 , 6.31 , 
        12.43 , 4.81, 9.28 , 4.82 , 3.85 , 6.88, 12.48, 11.66, 8.06, 5.97 ] 

y_data = [4.49, 4.12, 4.84, 4.93, 4.24, 4.9, 4.81, 4.57, 4.37, 4.61, 4.29, 
            4.55, 4.35, 3.56, 3.76, 4.18, 2.25, 4.46, 4.63, 4.27, 4.61, 4.11, 
            4.74, 3.98, 3.96, 4.42, 4.78, 4.55, 4.35, 4.25]


# alpha=3.7;
# beta =.71;

alpha , beta = 5.266713, 1.068839 
def func(alpha,beta,x_data):
    result=[]
    for x in x_data:
        y = alpha*x/(1+beta*x)
        result.append(y)
    return result

result = func(alpha,beta,x_data)

plt.scatter(x_data,result)
plt.scatter(x_data,y_data)
plt.show()

