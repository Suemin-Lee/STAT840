# genetic algorithm search for continuous function optimization
from numpy.random import randint
from numpy.random import rand
import numpy as np



x_data = [ 9.1, 4.41, 18.99, 13.73, 9.1, 30.89, 19.17, 8.64, 8.67, 15.62,
        14.68 , 11.09 , 11.53, 2.28 , 3.64, 5.5 , 0.73, 12.39, 25.7 , 6.31 , 
        12.43 , 4.81, 9.28 , 4.82 , 3.85 , 6.88, 12.48, 11.66, 8.06, 5.97 ] 

y_data = [4.49, 4.12, 4.84, 4.93, 4.24, 4.9, 4.81, 4.57, 4.37, 4.61, 4.29, 
            4.55, 4.35, 3.56, 3.76, 4.18, 2.25, 4.46, 4.63, 4.27, 4.61, 4.11, 
            4.74, 3.98, 3.96, 4.42, 4.78, 4.55, 4.35, 4.25]

def Q_func(theta):
	alpha = theta[0]
	beta  = theta[1]
	x_data = [ 9.1, 4.41, 18.99, 13.73, 9.1, 30.89, 19.17, 8.64, 8.67, 15.62,
	14.68 , 11.09 , 11.53, 2.28 , 3.64, 5.5 , 0.73, 12.39, 25.7 , 6.31 , 
	12.43 , 4.81, 9.28 , 4.82 , 3.85 , 6.88, 12.48, 11.66, 8.06, 5.97 ] 

	y_data = [4.49, 4.12, 4.84, 4.93, 4.24, 4.9, 4.81, 4.57, 4.37, 4.61, 4.29, 
				4.55, 4.35, 3.56, 3.76, 4.18, 2.25, 4.46, 4.63, 4.27, 4.61, 4.11, 
				4.74, 3.98, 3.96, 4.42, 4.78, 4.55, 4.35, 4.25]			
	result=[]
	for i,x in enumerate(x_data):
		y = y_data[i]
		least_square = alpha*x/(1+beta*x)
		result.append((y- least_square)**2)
	return sum(result)

# selection
def selection(pop, scores):
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), 2):
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two offspring
def crossover(p1, p2, r_cross):
	c1, c2 = p1.copy(), p2.copy()
	if rand() < r_cross:
		pt = randint(1, len(p1)-2)
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation 
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		if rand() < r_mut:
			bitstring[i] = 1 - bitstring[i] # flip the bit

# convert bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		chars = ''.join([str(s) for s in substring])
		integer = int(chars, 2)
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		decoded.append(value)
	return decoded

# genetic algorithm
def genetic_algorithm(likelihood_fun, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	best, best_eval = 0, likelihood_fun(decode(bounds, n_bits, pop[0]))
	for gen in range(n_iter):
		decoded = [decode(bounds, n_bits, p) for p in pop]
		scores = [likelihood_fun(d) for d in decoded]
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		selected = [selection(pop, scores) for _ in range(n_pop)]
		offspring = list()
		for i in range(0, n_pop, 2):
			p1, p2 = selected[i], selected[i+1]
			for c in crossover(p1, p2, r_cross):
				mutation(c, r_mut)
				offspring.append(c)
		pop = offspring
	return [best, best_eval]

# define range for input

bounds = [[4, 7],[.5,2]]
n_pop = 20    # define the population size
n_iter = 200   # define the total iterations
n_bits = 16    # bits per variable
r_cross = 0.1  # crossover rate
r_mut = 1.0 / (float(n_bits) * len(bounds))   # mutation rate

best, score = genetic_algorithm(Q_func, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score)) 


