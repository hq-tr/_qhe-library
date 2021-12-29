import numpy as np 
import misc
from scipy.sparse import *
from itertools import product, combinations_with_replacement, compress

def single_particle_state_disk(m):
	return lambda z: z**m * np.exp(-abs(z)**2) * np.sqrt(2**m/(2*np.pi))  / np.prod([np.sqrt(x) for x in range(1,m+1)])

def single_particle_state_disk_conj(m):
	return lambda z: np.conj(z)**m * np.exp(-abs(z)**2) * np.sqrt(2**m/(2*np.pi))  / np.prod([np.sqrt(x) for x in range(1,m+1)])

def single_particle_state(s, m):
	return lambda theta, phi: (np.cos(theta/2))**(s+m) * (np.sin(theta/2))**(s-m) * np.exp(1j*m*phi) / misc.sphere_coef(s,m)

def single_particle_state_conj(s, m):
	return lambda theta, phi: (np.cos(theta/2))**(s+m) * (np.sin(theta/2))**(s-m) * np.exp(-1j*m*phi) / misc.sphere_coef(s,m)

def sqfactorial(x):
	return np.prod(np.sqrt(np.arange(x)+1))

def density_element(Lam, Mu, No, theta, phi, bosonic=False):
	# Lam and Mu must be in index format
	S = (No-1)/2
	Ne = len(Lam)
	if len(Lam) != len(Mu):
		return 0
	elif Lam == Mu:
		term = sum([abs(single_particle_state(S,S-x)(theta,phi))**2 for x in Lam])
		return term
	elif len(Lam) == 1:
		return single_particle_state_conj(S,S-Lam[0])(theta, phi)*single_particle_state(S,S-Mu[0])(theta,phi)
	else:
		unq = find_unique_elements(Lam, Mu)
		if len(unq)==0:
			return 0
		else:
			#Lam_a = Lam_unique[0]
			#Mu_b  = Mu_unique[0]
			#print((Lam_a, Mu_b))
			Lam_a = unq[0][0]
			Mu_b  = unq[0][1]
			if bosonic: 
				mul_a = Lam.count(Lam_a)
				mul_b = Mu.count(Mu_b)
				mul_L = np.prod([sqfactorial(Lam.count(part)) for part in Lam])
				mul_M = np.prod([sqfactorial(Mu.count(part)) for part in Mu])
				return (mul_L*mul_M)*single_particle_state_conj(S,S-Lam_a)(theta, phi)*single_particle_state(S,S-Mu_b)(theta,phi)
			else:
				a     = Lam.index(Lam_a)
				b     = Mu.index(Mu_b)
				return (-1)**(abs(a-b))*(single_particle_state_conj(S,S-Lam_a)(theta, phi)*single_particle_state(S,S-Mu_b)(theta,phi))

def get_density(basis_list,coef_list,No, theta,phi, bosonic=False):
	dim = len(basis_list)
	D   = (0+0j)*phi*theta
	#print(dim)
	i = 1
	total = dim**2
	for (m,p) in product(range(dim),range(dim)):
		#print(f"{i}/{total}           ", end="\r")
		#print((basis_list[m], basis_list[p]), end = "\t")
		#print(basis_list[m]== basis_list[p], end = "\t")
		i +=1
		term = np.conj(coef_list[m])*coef_list[p]*density_element(basis_list[m], basis_list[p], No, theta, phi, bosonic=bosonic)
		D+=term
	#print(D) 

	return abs(D)

def density_element_disk(Lam, Mu, x, y, bosonic=False):
	# Lam and Mu must be in index format
	z = (x+1j*y)/2
	Ne = len(Lam)
	#print(z[0])
	if Lam == Mu:
		term = sum([abs(single_particle_state_disk(x)(z))**2 for x in Lam])
		return term
	if len(Lam) != len(Mu):
		return 0*z
	elif len(Lam) == 1:
		return single_particle_state_disk_conj(Lam[0])(z)*single_particle_state_disk(Mu[0])(z)
	else:
		#Lam_unique = list(set(Lam).difference(set(Mu)))
		#Mu_unique  = list(set(Mu).difference(set(Lam)))
		#if len(Lam_unique)!=1 or len(Mu_unique)!=1:
		#	return 0*z
		#else:
		unq = find_unique_elements(Lam, Mu)
		if len(unq)==0:
			return 0*z
		else:
			#Lam_a = Lam_unique[0]
			#Mu_b  = Mu_unique[0]
			#print((Lam_a, Mu_b))
			Lam_a = unq[0][0]
			Mu_b  = unq[0][1]
			if bosonic: 
				mul_L = np.prod([sqfactorial(Lam.count(part)) for part in Lam])
				mul_M = np.prod([sqfactorial(Mu.count(part)) for part in Mu])
				return (mul_L*mul_M)*single_particle_state_disk_conj(Lam_a)(z)*single_particle_state_disk(Mu_b)(z)
			else:
				a     = Lam.index(Lam_a)
				b     = Mu.index(Mu_b)
				return (-1)**(abs(a-b))*(single_particle_state_disk_conj(Lam_a)(z)*single_particle_state_disk(Mu_b)(z))

def density_matrix_disk(basis_list, pos):
	dim = len(basis_list)
	rho = dok_matrix((dim,dim), dtype=float)
	for (i,j) in combinations_with_replacement(range(dim), 2):
		term = density_element_disk(basis_list[i], basis_list[j], x,y)
		if abs(term)>1e-14:
			rho[i,j] += term
			if i!=j:
				rho[j,i]+= np.conj(term)

	return rho

def prune(partition,element):
	p_copy = [x for x in partition]
	p_copy.remove(element)
	return p_copy

def find_unique_elements(Lambda, Mu):
	L_copy = [x for x in Lambda]
	M_copy = [x for x in Mu]
	return list(compress(list(product(Lambda,Mu)), [prune(Lambda,x)==prune(Mu,y) for (x,y) in product(Lambda,Mu)]))

def get_density_disk(basis_list,coef_list, x,y, bosonic=False):
	dim = len(basis_list)
	#D   = (0+1j*0)*x*y
	#print(dim)
	#i = 1
	#total = dim**2\
	def get_element(param):
		#print(param, end = "\r")
		return np.conj(coef_list[param[0]])*coef_list[param[1]]*density_element_disk(basis_list[param[0]], basis_list[param[1]], x,y, bosonic=bosonic)
	D = sum(map(get_element, product(range(dim),range(dim))))

	return np.real(D)

def density_element_disk_function(Lam, Mu):
	# Lam and Mu must be in index format
	Ne = len(Lam)
	#print(z[0])
	if Lam == Mu:
		return lambda z: sum([abs(single_particle_state_disk(x)(z))**2 for x in Lam])
	if len(Lam) != len(Mu):
		return lambda z: 0*z
	elif len(Lam) == 1:
		return lambda z: single_particle_state_disk_conj(Lam[0])(z)*single_particle_state_disk(Mu[0])(z)
	else:
		Lam_unique = list(set(Lam).difference(set(Mu)))
		Mu_unique  = list(set(Mu).difference(set(Lam)))
		if len(Lam_unique)!=1 or len(Mu_unique)!=1:
			return lambda z: 0*z
		else:
			Lam_a = Lam_unique[0]
			Mu_b  = Mu_unique[0]
			#print((Lam_a, Mu_b))
			a     = Lam.index(Lam_a)
			b     = Mu.index(Mu_b)
			return lambda z: (-1)**(abs(a-b))*(single_particle_state_disk_conj(Lam_a)(z)*single_particle_state_disk(Mu_b)(z))

def get_density_disk_function(basis_list,coef_list):
	dim = len(basis_list)
	#D   = (0+1j*0)*x*y
	#print(dim)
	#i = 1
	#total = dim**2\
	def get_element(param):
		#print(param, end = "\r")
		return lambda z: np.conj(coef_list[param[0]])*coef_list[param[1]]*density_element_disk_function(basis_list[param[0]], basis_list[param[1]])(z)
	return lambda z: sum([f(z) for f in map(get_element, product(range(dim),range(dim)))])

def density_matrix_function(basis_list):
	dim = len(basis_list)
	Ne = len(basis_list[0])
	term = []
	row  = []
	col  = []
	for (i,j) in product(range(dim),range(dim)):
		Lam = basis_list[i]
		Mu  = basis_list[j]
		func = None
		if Lam == Mu:
			func = lambda z: sum([abs(single_particle_state_disk(x)(z))**2 for x in Lam])
		elif len(Lam) != len(Mu):
			continue
		elif len(Lam) == 1:
			func = lambda z: single_particle_state_disk_conj(Lam[0])(z)*single_particle_state_disk(Mu[0])(z)
		else:
			Lam_unique = list(set(Lam).difference(set(Mu)))
			Mu_unique  = list(set(Mu).difference(set(Lam)))
			if len(Lam_unique)!=1 or len(Mu_unique)!=1:
				continue
			else:
				Lam_a = Lam_unique[0]
				Mu_b  = Mu_unique[0]
				#print((Lam_a, Mu_b))
				a     = Lam.index(Lam_a)
				b     = Mu.index(Mu_b)
				func = lambda z: (-1)**(abs(a-b))*(single_particle_state_disk_conj(Lam_a)(z)*single_particle_state_disk(Mu_b)(z))
		if func != None:
			term.append(func)
			row.append(i)
			col.append(j)
			if i!=j:
				term.append(lambda z: np.conj(func(z)))
				row.append(j)
				col.append(i)
	return lambda z: coo_matrix(([foo(z) for foo in term], (row, col)), shape=(dim,dim))


if __name__=="__main__":
	import FQH_states as FQH
	state = FQH.fqh_state("state_laughlin")
	basis = state.get_basis_index()

	rho = get_density_disk_function(basis, state.coef)
	for i in range(5):
		print(rho(i))

