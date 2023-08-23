# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:25:30 2020

@author: HQT

This library contain functions that are useful for numerics on FQHE.
See <misc_readme.txt> for a table of content.

LAST UPDATED 2022-12-12
"""

from scipy import *
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import inv
from scipy.linalg import eig, eigh
import time
from scipy.linalg import null_space
import sys
import operator as op
from functools import reduce
from itertools import zip_longest, chain
from multiprocessing import Pool

def flip(x):
	if x=="1":
		return "0"
	elif x=="0":
		return "1"
	else:
		return ""

def PH_conj(binary_basis):
	return "".join(map(flip, binary_basis))

def display_vector(basis, coef):
	assert len(basis)==len(coef), "Basis and coefficients must be of the same dimension"
	for (x,y) in zip_longest(basis,coef):
		print(x,end="\t")
		print(y)
	return 

def index_to_binary(basis_index,Np,get_string=False,PH_conj=False):
	if PH_conj:
		one = "0"
		zer = "1"
	else:
		one = "1"
		zer = "0"
	if get_string:
		vec = ""
		for i in range(Np):
			if i in basis_index:
				vec=vec+"1"
			else:
				vec=vec+"0"
	else:
		if PH_conj:
			vec = ones(Np, dtype=int)
			vec[basis_index] = 0
		else:
			vec = zeros(Np, dtype=int)
			vec[basis_index] = 1
	return vec

def index_to_binary_boson(basis_index,Np,inc_space = False):
	if inc_space:
		return " ".join(list(map(str,[basis_index.count(x) for x in range(Np)])))
	else:
		return "".join(list(map(str,[basis_index.count(x) for x in range(Np)])))

def binary_to_index_boson(basis, invert = False, n = 0):
	if invert:
		basis_2 = basis.split()[::-1]
	else:
		basis_2 = basis.split()
	return list(chain.from_iterable([i-n]*int(basis_2[i]) for i in range(len(basis_2))))

def binary_to_index(basis, n=0):
	return [x-n for x in range(len(basis)) if basis[x]=="1" ]

def index_to_decimal_boson(basis, invert=False):
	basis_fermion = basis_b2f(basis)
	return index_to_decimal(basis_fermion)

def particle_hole_boson(basis_index,Np):
	Ne = len(basis_index)
	Nh = Np-Ne
	basis_hole = [0]*basis_index[0]
	for i in range(1,Ne):
		basis_hole+=[i]*(basis_index[i]-basis_index[i-1]-1)
	basis_hole += [Ne] * (Nh-len(basis_hole))
	
	return basis_hole

def binary_to_decimal_boson(basis, invert = False):
	basis_index = binary_to_index_boson(basis, invert=invert)
	basis_index_convert = basis_b2f(basis_index)
	return index_to_decimal(basis_index_convert)

def basis_b2f(basis_index):
	return [basis_index[i]+i for i in range(len(basis_index))]

def basis_f2b(basis_index):
	return [basis_index[i]-i for i in range(len(basis_index))]

def decimal_to_binary_boson(basisvec, Np, invert = False):
	return index_to_binary_boson(basis_f2b(dec_to_index(basisvec)), Np, inc_space=True)

def decimal_to_binary(basisvec, Np, invert = False):
	vec_index = dec_to_index(basisvec)
	if invert:
		return index_to_binary(vec_index, Np, get_string=True)[::-1]
	else:
		return index_to_binary(vec_index, Np, get_string=True)

def binary_to_decimal(basisvec, invert=False):
	if invert: 
		return int(basisvec, 2)
	else:
		return int(basisvec[::-1],2)


def index_to_decimal(basisvec):
	return np.sum(2**(np.array(basisvec)))

def dec_to_occupation(num,Np,get_array=True):
	if get_array:
		numi = dec_to_index(num)
		return array([x in numi for x in range(Np)], dtype = int)
	else:
		numb = bin(num)
		return [1 if numb[-i-1]=="1" else 0 for i in range(Np)]

def occupation_str_to_list(occupation, get_array=True):
	if get_array:
		return np.array(list(map(int,occupation)))
	else:
		return list(map(int,occupation))
	
def dec_to_index(num):
	numb = bin(num)
	return [x for x in range(len(numb)-2) if numb[-x-1]=="1"]
# =============================================================================
# def find_kernel(LL,iteration = 60): # Not in use
#     N = shape(LL)[0]
#     M = inv(LL.tocsc() - 0.0000001*identity(N,format = 'csc'))
#     vec = [None]*iteration
#     for n in range(iteration):
#         print(n,end = '.\r')
#         v0 = rand(N); v0 = v0/np.sqrt(np.sum(v0**2));
#         for i in range(1000):
#             v0 = M*v0
#             v0 = v0/np.sqrt(np.sum(v0**2))
#         vec[n] = v0
#     
#     M = empty([iteration, iteration])
#     for i in range(iteration):
#         for j in range(iteration):
#             M[i,j] = np.dot(vec[i],vec[j])
#     
#     E, V = eigh(M)
#     zerovec = V[abs(E)>0.000001]
# # =============================================================================
# #     M = zeros(iteration)
# #     for i in range(iteration):
# #         for j in range(iteration):
# #             M[i,j] = np.dot(vec[i],vec[j])
# #     
# #     EM, VM = []
# # =============================================================================
#     return zerovec           
# =============================================================================

# =============================================================================
# def get_vector(nameroot, dim, basisfile): 
#     vec     = [] 
#         
#     for i in range(dim):
#         f = open(nameroot+"_"+str(i),"r")
#         vec.append([float(x) for x in f.readlines()])
#         f.close()
#     
#     M = empty([dim,dim])
#     for i in range(dim):
#         for j in range(dim):
#             M[i,j] = np.dot(vec[i],vec[j])
#     
#     E,V = eig(M); V = V.T
#     print("Finished diagonalization.")
#     outnameroot = "output/" + str(input("Output name file: "))
#     j = 0
#     
#     f = open(basisfile, "r")
#     basis = f.readlines()
#     N = len(basis)
#     
#     for i in range(dim):
#         if E[i]>10e-8:
#             print("Energy: " + str(E[i]))
#             f = open(outnameroot+"_"+str(j),"w+")
#             f.write(str(N)+"\n")
#             coef = np.dot(V[i], vec)
#             for k in range(N):
#                 f.write(basis[k])
#                 f.write(str(coef[k])+"\n")
#             j+=1
#     print("Successfully saved " + str(j) + " files.")
#     return
# =============================================================================

def record(fname, *args):
	num = len(args[1])
	with open(fname, "w+") as f:
		f.write("\n".join(" ".join(str(x[i]) for x in args) for i in range(num)))
	return

def print_matrix_to_file(dim, LL):
	fileopen = str(input("Enter file name: "))
	f = open(fileopen, "w+")
	f.write(str(dim) + '\n')
	for i in range(dim):
		f.write(str(LL[i,i])+'\n')
	f.write('-1\n')
	
	for i in range(dim):
		for j in range(i+1,dim):
			if LL[i,j]>0:
				f.write(str(LL[i,j])+" "+str(j)+" ")
		f.write("0 0\n")
	f.write('-1 -1\n')
	
	for j in range(dim):
		for i in range(j):
			if LL[i,j]>0:
				f.write(str(i)+ " ")
		f.write("-1\n")
	
	f.close()
	print('File saved successfully.')
	return

def print_basis_to_file(Ne, Np, basis_index, filename=None, invert=False):
	if filename==None:
		filename=str(input("Enter basis output file name: "))
	fileopen = "basis_files/"+filename
	f = open(fileopen, "w+")
	dim  = len(basis_index)
	f.write(str(Ne) + " " +str(Np)+" " + str(dim) + "\n")
	for vec in basis_index:
		if invert:
			towrite = " ".join(list(map(str,[Np-1-x for x in vec[::-1]])))
		else:
			towrite = " ".join(list(map(str,vec)))
		f.write(towrite)
# =============================================================================
#         for number in vec:
#             if invert:
#                 f.write(str(Np-number-1)+ " ")
#             else:
#                 f.write(str(number) + " ")
# =============================================================================
		f.write(" -1\n")
	f.close()
	print('File saved successfully.')
	return

def collate_vectors(basis1, coef1, basis2, coef2,get_array = False):
	dict1 = {basis1[i]: coef1[i] for i in range(len(basis1))}
	dict2 = {basis2[i]: coef2[i] for i in range(len(basis2))}
	basis = list(set(basis1).union(set(basis2)))
	if get_array:
		coef_1 = np.array([dict1[vec] if vec in dict1 else 0.0 for vec in basis])
		coef_2 = np.array([dict2[vec] if vec in dict2 else 0.0 for vec in basis])
	else:
		coef_1 = [dict1[vec] if vec in dict1 else 0.0 for vec in basis]
		coef_2 = [dict2[vec] if vec in dict2 else 0.0 for vec in basis]
	return basis, coef_1, coef_2

def collate_many_vectors(basis_list, coef_list):
	assert len(basis_list) == len(coef_list), "Dimension mismatch"
	num = len(basis_list)
	if num == 0:
		return
	if num == 1:
		return basis_list[0], coef_list[0]
	elif num == 2:
		bas, coef1, coef2 = collate_vectors(basis_list[0], coef_list[0], basis_list[1], coef_list[1])
		return bas, np.array([coef1,coef2])
	else:
		vec = []
		for i in range(num):
			assert len(basis_list[i]) == len(coef_list[i]), f"Dimension mismatch at vector #{i+1}"
			dim = len(basis_list[i])
			vec.append({basis_list[i][x]: coef_list[i][x] for x in range(dim)})
		total_basis = list(set.union(*map(set, basis_list)))
		new_coef    = np.zeros((num, len(total_basis)))
		for i in range(num):
			new_coef[i] = [vec[i][x] if x in vec[i] else 0.0 for x in total_basis]

		return total_basis, new_coef


def read_basis_from_file(filename = None, outdec = False, invert = False):
	if filename == None:
		filename = str(input("Enter basis input file name: "))
	try:
		with open(filename, "r") as f:
			a = [x.split() for x in f.readlines()]
	except FileNotFoundError:
		print("File not found.")
		return
	
	print("Ne = " + a[0][0])
	print("No = " + a[0][1])
	print("The dimension is " + a[0][2])
	
	outvec = []
	if invert:
		No = int(a[0][1])
		if outdec:
			for vec in a[1:]:
				outvec.append(index_to_decimal([No-1-int(x) for x in vec[:-1]]))
		else:        
			for vec in a[1:]:
				outvec.append([No-1-int(x) for x in vec[:-1]])
	else:
		if outdec:
			for vec in a[1:]:
				outvec.append(index_to_decimal([int(x) for x in vec[:-1]]))
		else:        
			for vec in a[1:]:
				outvec.append([int(x) for x in vec[:-1]])
	
	Np = int(a[0][1])
	return outvec, Np

def read_mat_from_file(fileopen=None):
	if fileopen==None:
		fileopen = str(input("Enter matrix file name: "))
	if (fileopen == "quit"):
		return
	try:
		f = open(fileopen, "r")
	except FileNotFoundError:
		print("File not found.")
		return
	a = [x.split() for x in f.readlines()]
	d = int(a[0][0])
	print("The dimenstion is " + str(d))
	M = dok_matrix((d,d), dtype = float)
	for x in a[1:]:
		M[int(x[0]),int(x[1])] = float(x[2])
		M[int(x[1]),int(x[0])] = float(x[2])
	f.close()
	print("Finished importing matrix")
	return M.tocsc()

def truncate_or(vec,n1,m1,n2,m2):
	if ((len([i for i in vec if i<n1])<=m1 and len([i for i in vec if i>Np-n1-1])<=m1)) or (( len([i for i in vec if i<n2])<=m2 and len([i for i in vec if i>Np-n2-1])<=m2)):
		return True
	else:
		return False

def truncate_from_vector(basisfile=None, coeffile=None, get_basis = True):
	n1 = 2
	m1 = 1
	n2 = 5
	m2 = 2
	if basisfile==None:
		basisfile = str(input("Enter basis file name: "))
	try:
		f = open(basisfile,"r")
		a = [list(map(int,x.split())) for x in f.readlines()]
		f.close()
	except FileNotFoundError:
		print("Basis file not found")
		return
	
	if coeffile==None:
		coeffile = str(input("Enter coefficient file: "))
	try:
		f = open(coeffile,"r")
		b = array([float(x) for x in f.readlines()])
		f.close()
	except FileNotFoundError:
		print("Coefficient file not found")
		return
	
	outfile = str(input("Enter output file name: "))
	f = open(outfile, "w+")
	
	dim = a[0][2]
	Np = a[0][1]

	print("The number of electrons is "+str(a[0][0]))
	print("The number of orbitals is " + str(Np))
	print("The dimension is " + str(dim))
	
	for j in range(dim):
		vec = a[j+1][:-1]
		if not ((len([i for i in vec if i<n1])<=m1 and len([i for i in vec if i>Np-n1-1])<=m1) or ( len([i for i in vec if i<n2])<=m2 and len([i for i in vec if i>Np-n2-1])<=m2)):
			b[j] = inf
			
	b = b/np.sum(b[b<inf]**2) # Normalize
	tdim = np.sum(b<inf)
	print("The truncated dimension is " + str(tdim))
	if get_basis:
		f.write(f"{tdim}\n")
		for i in range(dim):
			if b[i]<inf:
				vec = a[i+1][:-1]
				f.write(index_to_binary(vec,Np,True)+"\n")
				f.write(str(b[i])+"\n")
	else:
		for co in b:
			if (co<inf):
				f.write(str(co)+"\n")
	f.close()
	print("File saved successfully")
			
	return

def index_to_boson(basis,Np): 
	Ne = len(basis)
	thing = [-1]+basis
	occ_list = [thing[x+1]-thing[x]-1 for x in range(Ne)]
	return "".join(map(str,occ_list))+str(Np-basis[-1]-1)


def gram_schmidt(veclist):
	orthovec = []
	checkvec = array(veclist[0])
	orthovec.append(checkvec/np.sqrt(np.sum(checkvec*checkvec)))
	i=1
	for vec in veclist[1:]:
		coef = np.dot(orthovec, vec)
		checkvec = vec-np.dot(coef, orthovec)
		norm = np.sum(checkvec*checkvec) # assume real co-efficients
		if norm > 1e-10:
			orthovec.append(checkvec/np.sqrt(norm))
		i+=1
	
	#if len(orthovec) == len(veclist):
		#print("Warning: Input basis might not be complete.")
	return orthovec

def read_multi_vector(name=None, N=None, startindex=0):
	if name==None:
		name = str(input("Enter file name root: "))
	if N==None:
		N = int(input("How many states to read? "))
	
	if N>1:
		with open(f"{name}0","r") as f:
			fcontent = array([float(x) for x in f.readlines()])
		dim = len(fcontent)
		vec = empty((N,dim))
		vec[0,:] = fcontent
		del fcontent
		missingno = []
		for i in range(N-1):
			#print("Reading vector "+str(i+1),end="\r")
			try:
				f = open(name+str(i+1), "r")
				vec[i+1,:] = array([float(x) for x in f.readlines()])
				f.close()
			except Exception as errormessage:
				missingno.append(i+1)
				print(f"Error: {errormessage} at file indexed {i+1}. Skipped.")

		vec = delete(vec,tuple(missingno), axis=0)
		print(f"{N-len(missingno)} files found.")
	else:
		f = open(name+"_"+str(i), "r")
		vec = array([float(x) for x in f.readlines()])
		f.close()
	return vec

def print_multi_vector(vec, name=None):
	if name==None:
		name = str(input("Enter file name root: "))
	N = len(vec)
	for i in range(N):
		f = open(name+"_"+str(i), "w+")
		for number in vec[i]:
			f.write(f"{np.real(number)}\n")
			if np.imag(number)>1e-8:
				print("Warning: non-real co-efficients")
		f.close()
	print("Files saved successfully.")
	return

def sandwich(a,M,b, M_is_sparse=True):  # Calculate <a|M|b>
	if M_is_sparse:
		return np.dot(a,M.tocsc().np.dot(b))
	else:
		return np.dot(a,np.dot(M,b))

def read_profYang_files(name=None, N=None, startindex = None, basisformat = "index", sortbasis=False, noti=True):
	if name==None:
		name = str(input("Enter file name root: "))
	
	if N==None:
		N = int(input("How many states to read? "))
		
	if N>1:
		if startindex == None:
			startindex = int(input("Starting index: "))
			
		# Read the basis and convert to either decimal or index format
		with open(name+str(startindex), "r") as f:
			dim = int(f.readline())
			if basisformat == "dec":
				basis = [int(x) for x in f.readlines()[::2]]
				if sortbasis:
					basis.sort()
			else:
				a = [int(x) for x in f.readlines()[::2]]
				if sortbasis:
					a.sort()
				b = [bin(x) for x in a]
				basis = []
				for num in b:
					basis.append([x-1 for x in range(1,len(num)-1) if num[-x]=="1"])
				del a,b

		if noti: print("The dimension is "+str(dim))
		
		# Read the co-efficients    
		vec = empty([N,dim])
		missingno = []
		for i in range(startindex, startindex+N):
			print("Reading vector "+str(i+1-startindex)+" out of " + str(N),end="\r")
			try:
				with open(name+str(i), "r") as f:
					dim = int(f.readline())
					a = f.readlines()
					b = array([complex(float(a[2*i]),float(a[2*i+1])) for i in range(dim)])
					if sortbasis:
						vec[i-startindex] = np.imag(sort(b))
					else:
						vec[i-startindex] = np.imag(b)
			except FileNotFoundError:
				missingno.append(i-startindex)
				print("File number "+str(i)+" not found. Skipped.")
		
		vec = delete(vec, tuple(missingno),axis=0)
		print(str(N-len(missingno))+" files found")
	else:
		f = open(name, "r")
		a = f.readlines()
		f.close()
		
		dim = int(a[0])
		if basisformat == "dec":
			basis = [int(x) for x in a[1::2]]
		else:
			b = [bin(int(x)) for x in a[1::2]]
			
			basis = []
			for num in b:
				basis.append([x-1 for x in range(1,len(num)-1) if num[-x]=="1"])
		vec = [float(x) for x in a[2::2]]
		
	return basis, vec, dim

def read_basis_from_vector(vecname = None, basisformat = "bin"):
	if vecname==None:
		vecname = str(input("Enter vector file name: "))
		
	f = open(vecname,"r")
	a = f.readlines()[1::2]
	f.close()
	if basisformat == "bin":
		return [[int(x) for x in y[:-1]] for y in a]
	elif basisformat == "index":
		N = len(a[0])
		return [[x for x in range(N) if y[x]=="1"] for y in a]
	elif basisformat == "binstr":
		return [x[:-1] for x in a]
	elif basisformat == "dec":
		Np = len(a[0])-1
		return [int(x[-1::-1],2) for x in a]

def extract_LEC_states(basisphi,basispsi,Np,LEC=[2,1,5,2],getfull = False, complement=False): # extract LEC sub-Hilbert space from given Hilbert space
	# basisphi is the occupational number basis
	# basispsi is the orthonormal basis of the full Hilbert space expressed as coefficients of vectors in basisphi
	# basispsi and basisphi must follow the same order
	LECbasis = []
	LECindex = [] 
	M        = []
	
	#LEC condition:
	n1 = LEC[0]
	m1 = LEC[1]
	n2 = LEC[2]
	m2 = LEC[3]
	
	flipBool = lambda x: not x
	idBool   = lambda x: x
	
	if complement:
		func = flipBool
	else:
		func = idBool
	#basispsi = array(basispsi)
	for i in range(len(basisphi)):
		vec = basisphi[i]
		if func(((len([i for i in vec if i<n1])<=m1 or len([i for i in vec if i<n2])<=m2) and (len([i for i in vec if i>Np-n1-1])<=m1 or len([i for i in vec if i>Np-n2-1])<=m2))):
			LECbasis.append(vec)
			LECindex.append(i)
		else:
			M.append(basispsi[:,i])
			
	print("There are " + str(len(LECbasis))+" LEC basis states.")
	M = array(M)
	#print(shape(M))        
	
	M_sq = np.dot(M.T,M)
	#print(shape(M_sq))
	
	#E,V = eig(M_sq)
	#V = V.T
	#print(sort(E)[:20])
	
	# LEC:
	#nullveccoef = V[abs(E)<1e-11]
	V = null_space(M_sq)
	nullveccoef = V.T
	
	nullvec = np.dot(nullveccoef, basispsi)
	
	# normalize
	#for i in range(len(nullveccoef)):
		#nullvec[i] = nullvec[i]/np.sqrt(np.dot(nullvec[i],nullvec[i]))
	
	if len(nullvec)>0:
		#nullvecout = array(gram_schmidt(nullvec))
		if getfull:
			return basisphi, nullvec
		else:
			return LECbasis, nullvec[:,LECindex]
	else:
		return [], []
	

		
def check_overlap(vecfile1, vecfile2=None):
	if vecfile2==None: vecfile2=vecfile1
	f = open(vecfile1, 'r')
	a = f.readlines()
	f.close()
	dim1 = int(a[0])
	b = [complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)]
	sort(b)
	coef1 = np.imag(b)
	
	f = open(vecfile2, 'r')
	a = f.readlines()
	dim2 = int(a[0])
	if dim1==dim2:
		b = [complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)]
		sort(b)
		coef2 = np.imag(b)
		return (np.dot(coef1,coef2))**2
	else:
		print("Dimension mismatch.")
		return 0

def dec_to_bin_file(infile=None, outfile=None, Np=None):
	if infile==None:
		infile = str(input("Input file name: "))
	if outfile== None:
		outfile = str(input("Output filename: "))
	if Np == None:
		Np = int(input("Number of orbitals: "))
	
	basis, vec, dim = read_profYang_files(infile, N=1, startindex = None)
	
	f = open(outfile, "w+")
	f.write(f"{dim}\n")
	toprint = "".join(f"{index_to_binary(Np-1-array(basis[j]), Np, True)}\n{np.real(vec[j]):.12f}\n" for j in range(dim))
	f.write(toprint)
	f.close()
	return

def bin_to_dec_file(infile=None, outfile=None, Np=None, invert = False):
	if infile==None:
		infile = str(input("Input file name: "))
	if outfile== None:
		outfile = str(input("Output filename: "))
	if Np == None:
		Np = int(input("Number of orbitals: "))
	
	f = open(infile, "r")
	a = [x.split()[0] for x in f.readlines()]
	f.close()
	dim = int(a[0])
	
	toprint = f"{dim}\n"
	if not invert:
		toprint +="".join(f"{int(a[i][::-1],2)}\n{a[i+1]}\n" for i in arange(1,2*dim+1,2,dtype=int))
	else:
		toprint +="".join(f"{int(a[i],2)}\n{a[i+1]}\n" for i in arange(1,2*dim+1,2,dtype=int))
	f = open(outfile,"w+")
	f.write(toprint)
	f.close()
	return dim

def bin_to_dec_file_full(basislist, infile=None, outfile=None, invertbas = False, invertvec = False, coef_only = False):
	#if basfile==None:
	#    basfile = str(input("Input basis file name: "))
	if infile==None:
		infile = str(input("Input file name: "))
	if outfile== None:
		outfile = str(input("Output filename: "))
	
	#basislist, Np = read_basis_from_file(basfile, True, invertbas)
	
	f = open(infile, "r")
	a = [x.split()[0] for x in f.readlines()]
	f.close()
	tdim = int(a[0])
	dim = len(basislist)
	if invertvec:
		b = {int(a[i][::-1],2): a[i+1] for i in arange(1,(2*tdim+1),2)}
	else:
		b = {int(a[i],2): a[i+1] for i in arange(1,(2*tdim+1),2)}
	
	if coef_only:
		toprint = "".join(f"{b[vec]}\n" if (vec in b) else f"0.000000\n" for vec in basislist)
	else:
		toprint = f"{dim}\n"+"".join(f"{vec}\n{b[vec]}\n" if (vec in b) else f"{vec}\n0.000000\n" for vec in basislist)
	
	f = open(outfile,"w+")
	f.write(toprint)
	f.close()
	return tdim

def truncated_to_full_file(basislist, infile=None, outfile=None, invertbas = False, invertvec = False, coef_only = False):
	#if basfile==None:
	#    basfile = str(input("Input basis file name: "))
	if infile==None:
		infile = str(input("Input file name: "))
	if outfile== None:
		outfile = str(input("Output filename: "))
	
	#basislist, Np = read_basis_from_file(basfile, True, invertbas)
	
	f = open(infile, "r")
	a = [x.split()[0] for x in f.readlines()]
	f.close()
	tdim = int(a[0])
	basis = [int(x) for x in a[1::2]]
	coef  = array([float(x) for x in a[2::2]])
	coef  = coef/np.sqrt(np.dot(coef,coef))
	dim = len(basislist)
	if invertvec:
		print("LOL I haven't written this portion of the code")
		return
	else:
		b = {x:y for [x,y] in zip_longest(basis,coef)}
	
	if coef_only:
		toprint = "".join(f"{b[vec]}\n" if (vec in b) else f"0.000000\n" for vec in basislist)
	else:
		toprint = f"{dim}\n"+"".join(f"{vec}\n{b[vec]}\n" if (vec in b) else f"{vec}\n0.000000\n" for vec in basislist)
	
	f = open(outfile,"w+")
	f.write(toprint)
	f.close()
	return tdim

def get_full_basis_vector(basisname=None,targetname=None):
	if basisname==None:
		basisname = str(input("Enter basis file name: "))
	if targetname==None:
		targetname = str(input("Enter basis vector file names: "))
	
	basis, Np = read_basis_from_file(f"basis_files/{basisname}", outdec=True)
	dim = len(basis)
	for i in range(dim):
		toprint = f"{dim}\n"
		toprint += "".join(f"{basis[j]}\n1.00\n" if j==i else f"{basis[j]}\n0.00\n" for j in range(dim))
		f = open(f"basis_files/vectors/{targetname}_{i}","w+")
		f.write(toprint)
		f.close()
	print("Files saved.")
	return

def findLZ(num,S,debug=False): # determine Lz sector of vector represented by decimal number num
	a = bin(num)[2:]
	ne = 0
	b_ind = []
	for i in range(len(a)):
		if a[-i-1] =='1':
			ne+=1
			b_ind.append(i)
	if debug: print(f"{num}\tThe number of electrons is "+str(ne))
	Lz = np.sum(b_ind)-ne*S
	#print("The Lz sector is "+str(Lz))
	return Lz

def orthonormalize(vec, tol=1e-12):
	M = np.dot(vec,vec.T)
	E,V = eig(M)
	#print(E)
	orthocompo = V[:,abs(E)>tol]
	#f = open("test","w+")
	#towrite = "".join(f"{x}\n" for x in sort(E))
	#f.write(towrite)
	#f.close()  
	del V,E
	vecout = np.dot(orthocompo.T,vec)
	for i in range(len(vecout)):
		vecout[i] = vecout[i]/np.sqrt(np.dot(vecout[i],vecout[i]))
	return vecout

def orthonormalize_basis(inrootname=None, N=None,outrootname=None, savefile=True):
	if inrootname==None:
		inrootname = str(input("Input file name root: "))
	if N==None:
		N = int(input("Number of input vectors: "))
	if outrootname==None:
		outrootname = str(input("Output file name root: "))
	
	vec = read_multi_vector(inrootname, N)
	if len(vec)==0:
		print("Process terminated.")
		return
	
	orthovec = orthonormalize(vec)
	
	ran = len(orthovec)
	print(str(ran)+" orthonormal vectors.")
	
	if savefile:
		for i in range(ran):
			vecprint = np.real(orthovec[i]/np.sqrt(np.dot(orthovec[i],orthovec[i])))
			towrite = "".join(f"{x:.16f}\n" for x in vecprint)
			f = open(f"{outrootname}_{i}","w+")
			f.write(towrite)
			f.close()
		
		print("Files saved successfully.")
		return
	else:
		return orthovec

def orthonormalize_profYang_basis(inrootname=None, N=None,outrootname=None, savefile=True, tol=1e-10, GS=False):
	if inrootname==None:
		inrootname = str(input("Input file name root: "))
	if N==None:
		N = int(input("Number of input vectors: "))
	if outrootname==None:
		outrootname = str(input("Output file name root: "))
	
	basis,vec,dim = read_profYang_files(inrootname, N=N,startindex=0)
	if len(vec)==0:
		print("Process terminated.")
		return

	if GS:
		orthovec = gram_schmidt(vec)
	else:
		orthovec = orthonormalize(vec, tol=tol)
	
	ran = len(orthovec)
	print(str(ran)+" orthonormal vectors.")
	
	if savefile:
		for i in range(ran):
			fname = outrootname+str(i)
			print_vector(basis,orthovec[i],fname,noti=False)
			print("Saving file "+str(i+1)+" out of "+str(ran), end="\r")
		
		print("Files saved successfully.")
		return
	else:
		return basis, orthovec

def read_dimensions():
	f = open("master_files/dimensions.txt","r")
	a = [[float(x) for x in y.split()] for y in f.readlines()[1:]]
	f.close()
	return {(x[0],x[1],x[2]): list(map(int,x[3:])) for x in a}

def remove_projection(vecfile1=None, vecfile2=None, outputname=None, getvec=False):
	if vecfile1 == None:
		vecfile1 = str(input("Base vector file name: "))
	if vecfile2 == None:
		vecfile2 = str(input("Subtracted vector file name: "))
	if outputname == None:
		outputname = str(input("Output file name: "))
		
	f = open(vecfile1, "r")
	a = f.readlines()
	f.close()
	dim1 = int(a[0])
	b1 = [complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)]
	sort(b1)
	basis1 = np.real(b1)
	coef1  = np.imag(b1)
	
	f = open(vecfile2, "r")
	a = f.readlines()
	f.close()
	dim2 = int(a[0])
	if dim1 == dim2:
		b2 = [complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)]
		sort(b2)
		coef2  = np.imag(b2)
		coef_out = coef1 - coef2*np.dot(coef1,coef2)
		coef_out = coef_out/np.sqrt(np.dot(coef_out,coef_out)) # Re-normalize
		f = open(outputname, "w+")
		f.write(f"{dim1}\n")
		towrite = "".join(f"{int(basis1[i])}\n{coef_out[i]:.16f}\n" for i in range(dim1))
		f.write(towrite)
		f.close()
		print("File saved successfully!")
		if getvec:
			return basis1, coef_out
		else:
			return
	else:
		print("Dimensions are not the same.\nProcess terminated.")
		return

def project_state(vecfile1=None, vecfile2=None, N=None, outputname=None,getvec=False):
	if vecfile1 == None:
		vecfile1 = str(input("Base vector file name: "))
	if vecfile2 == None:
		vecfile2 = str(input("Projection space basis file name root: "))
	if N == None:
		N = int(input("Projection space dimension: "))
	if outputname == None:
		outputname = str(input("Output file name: "))
		
	f = open(vecfile1, "r")
	a = f.readlines()
	f.close()
	dim1 = int(a[0])
	b1 = [complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)]
	sort(b1)
	basis1 = np.real(b1)
	coef1  = np.imag(b1)
	
	proj_vec = empty(dim1)
	
	for i in range(N):
		f = open(f"{vecfile2}{i}","r")
		a = f.readlines()
		dim2 = int(a[0])
		if dim2!=dim1:
			print("Dimension of vector number "+str(i)+" does not match dimension of base vector.")
			print("Process terminated")
			return
		b1 = [complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)]
		sort(b1)
		coef2 = np.imag(b1)
		proj_vec += coef2*np.dot(coef1,coef2)
	
	proj_vec = proj_vec/np.sqrt(np.dot(proj_vec,proj_vec)) # re-normalize
	f = open(outputname, "w+")
	f.write(f"{dim1}\n")
	towrite = "".join(f"{int(basis1[i])}\n{proj_vec[i]:.16f}\n" for i in range(dim1))
	f.write(towrite)
	f.close()
	print("File saved successfully!")
	if getvec:
		return basis1, proj_vec
	else:
		return

def ncr(n, r):
	r = min(r, n-r)
	numer = reduce(op.mul, range(n, n-r, -1), 1)
	denom = reduce(op.mul, range(1, r+1), 1)
	return numer / denom

def approximate_dimension(m,k,N):
	pmax = int(N/m)+1
	s = 0
	for p in range(pmax):
		s+=(-1)**p * ncr(k,p) * ncr(k+N-m*p-1, N-m*p)
	return s/np.prod(arange(k)+1)

def approximate_hw_dimension(m,k,N):
	pmax = int(N/m)+1
	s = 0
	for p in range(pmax):
		s+=(-1)**p * ncr(k,p) * ncr(k+N-m*p-1, N-m*p+1)
	return s/np.prod(arange(k)+1)

def remove_projection_subspace(vecfile1=None, N=None, vecfile2=None, outputname=None):
	if vecfile1 == None:
		vecfile1 = str(input("Base vector file name: "))
	if N == None:
		N = int(input("How many vectors? "))
	if vecfile2 == None:
		vecfile2 = str(input("Subtracted vector file name: "))
	if outputname == None:
		outputname = str(input("Output file name: "))
	
	f = open(vecfile2, "r")
	a = f.readlines()
	f.close()
	dim1 = int(a[0])
	b1 = array([complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)])
	sort(b1)
	basis1 = np.real(b1)
	coef1  = np.imag(b1)
	
	vec = empty((N,dim1))
	for i in range(N):
		f = open(f"{vecfile1}_{i}","r")
		a = f.readlines()
		f.close()
		dim2 = int(a[0])
		if dim2 != dim1:
			print("Dimension mismatch at vector number " + str(i))
			if getvec:
				return list()
			else:
				return
		b2 = array([complex(float(a[i]),float(a[i+1])) for i in arange(1,2*dim1,2)])
		sort(b2)
		basis2 = np.real(b2)
		if (basis1 != basis2).all():
			print("Basis mismatch at vector number " +str(i))
			if getvec:
				return list()
			else:
				return
		coef2 = np.imag(b2)
		v = coef2 - np.dot(coef1, coef2)*coef1
		vec[i] = v/np.sqrt(np.dot(v,v))
	
	vecout = orthonormalize(vec)
	if outputname=="x":
		return vecout
	else:
		N_ = len(vecout)
		for i in range(N_):
			f = open(f"{outputname}_{i}","w+")
			f.write(f"{dim1}\n")
			towrite = "".join(f"{basis1[j]}\n{vecout[i,j]:.16f}\n" for j in range(dim1))
			f.write(towrite)
			f.close()
		print(str(N_)+" files saved successfully")
		return

def print_vector(basis, coef=[], filename=None, basisformat = "dec", Np = None, noti = True,normalize=True, print_zero_coef = True):
	if filename==None:
		filename = str(input("Output file name: "))
	
	dim = len(basis)
	if len(coef)==0:
		coef = zeros(dim)
	elif normalize:
		norm = np.sqrt(np.dot(coef,coef))
		#print(norm)
		if norm < 1e-12:
			print(f"***Warning: zero-norm vector. Norm = {norm}")
			norm = 1
		coef /= norm
	
	if len(coef)!=dim:
		print("Dimension mismatch. Dimension of vector: " +str(dim)+". Dimension of coefficient: "+str(len(coef)))
		print("Process terminated")
		return
	dim_t = len(coef[coef>1e-12])
	if basisformat == "dec":
		if print_zero_coef:
			towrite = f"{dim}\n"+"".join(f"{index_to_decimal(basis[i])}\n{np.real(coef[i]):.16f}\n" for i in range(dim))
		else:
			norm_t = np.sqrt(np.dot(coef[coef>1e-12],coef[coef>1e-12]))
			coef/=norm_t
			towrite = f"{dim_t}\n"+"".join(f"{index_to_decimal(basis[i])}\n{np.real(coef[i]):.16f}\n" for i in range(dim) if np.real(coef[i])>1e-12)
	elif basisformat == "bin":
		if Np==None:
			Np = int(input("How many orbital? "))
		towrite = f"{dim}\n"+"".join(f"{index_to_binary(basis[i],Np,True)}\n{np.real(coef[i]):.16f}\n" for i in range(dim))
	with open(filename,"w+") as f: 
		f.write(towrite)
	if noti:
		print("Files saved successfully")
	return

def print_vector_boson(basis, coef=[],fname=None, Np=None):
	dim = len(basis)
	if len(coef)==0: coef = [0]*dim
	if len(coef)!=dim:
		print("Dimension mismatch. Dimension of vector: " +str(dim)+". Dimension of coefficient: "+str(len(coef)))
		print("Process terminated")
		return
	if Np==None:
		Np = int(input("Input n_orb: "))
	if fname==None:
		fname = str(input("Output file name: "))
	toprint = "".join(f"{index_to_binary_boson(basis[i],Np,inc_space=True)}\n{coef[i]:.16f}\n" for i in range(dim))
	with open(fname, "w+") as f:
		f.write(f"{dim}\n")
		f.write(toprint)
	return 

def read_plot_data(filename=None):
	if filename==None:
		filename = str(input("Input file name: "))
	f = open(filename)
	a = f.readlines()
	f.close()
	k = array([float(x.split()[0]) for x in a])
	ov1 = array([float(x.split()[1]) for x in a])
	ov2 = array([float(x.split()[2]) for x in a])
	return k, np.sqrt(ov1), np.sqrt(ov2),np.sqrt(ov1+ov2)

def is_admissible(vector, k,r):
	# check if in vector, for every consecutive k orbitals, there are no more than r particles
	N = len(vector)
	for i in range(N-k+1):
		if np.sum(vector[i:(i+k)])>r:
			return False
	return True

def isAdmissible(vector):
    global k,r
    # check if in vector, for every consecutive r orbitals, there are no more than k particles
    N = len(vector)
    for i in range(N-r+1):
        if vector[i:(i+r)].count("1")>k:
            return 0
    return 1
    
def is_admissible_string(root, k, r):
	N = len(root)
	for i in range(N-k+1):
		if root[i:(i+k)].count("1")>r:
			return False
	return True

def read_basis_from_vector_m123(filename=None, basisformat = "occupation"): #specialized function to address bugs in m123
	if filename == None:
		filename = str(input("Enter file name: "))
	f  = open(filename,"r")
	a = f.readlines()
	dim = int(a[0])
	print("The dimension is"+str(dim))
	if basisformat == "occupation":
		a1 = [[int(x) for x in y[:-1]] for y in a[1::3]]
		a2 = [[int(x) for x in y[:-1]] for y in a[2::3]]
		vec = [a1[i]+a2[i] for i in range(len(a1))]
	elif basisformat == "dec":
		vec = [int((a[3*i+1][:-1]+a[3*i+2][:-1])[::-1],2) for i in range(dim)]
	elif basisformat == "binstr":
		a1 = [y[:-1] for y in a[1::3]]
		a2 = [y[:-1] for y in a[2::3]]
		vec = [a1[i]+a2[i] for i in range(len(a1))]   
	#print(vec[:3])
	return vec

def filter_basis(basislist, inbasfile = None, incoeffile = None, outfile = None, inorder = True):
	if inbasfile == None:
		inbasfile = str(input("Input basis file: "))
	if incoeffile == None:
		incoeffile = str(input("Input coefficient file: "))
	if outfile == None:
		outfile = str(input("Output file name: "))
	with open(inbasfile) as f, open(incoeffile) as g:
		basfromfile  = [int(x) for x in f.readlines()] # Basis must be in decimal format
		coeffromfile = [float(x) for x in g.readlines()] 
	#f = open(outfile,"w+")
	co = []
	bs = []
	for [x,y] in zip_longest(basfromfile, coeffromfile):
		print(x,end="\t")
		print(y,end="\t")
		if x in basislist: # I also assume the order of appearance of basislist is the same as basfromfile
			co.append(y)
			if not inorder:
				bs.append(x)
			#basislist.remove(x) # This is a bad idea but it will reduce the time under the right conditions
			print("recorded_______________",end="\r",flush=1)
		else:
			print("removed______",end="\r",flush=1)
	
	f = open(outfile,"w+")
	towrite = "".join(f"{x}\n" for x in co)
	f.write(towrite)
	f.close()
	if not inorder:
		g = open("basis_ordered.txt","w+")
		towrite = "".join(f"{x}\n" for x in bs)
		g.close()
	f.close()
	return

def find_L2(matfilename=None, vecfilename=None):
	if matfilename==None: matfilename = str(input("Input matrix file name: "))
	if vecfilename==None: vecfilename = str(input("Input matrix file name: "))
	LL  = read_mat_from_file(matfilename)
	with open(vecfilename) as f:
		vec = [float(x) for x in f.readlines()[2::2]]
	return sandwich(vec,LL,vec)

def L_plus(vecfile=None, N=1): #Non-functional
	basis, vec, dim = read_profYang_files(vecfile, N, startindex=0)
	return 

def invert_vec(filename=None,Np=None):
	if filename==None: filename = str(input("Input file name: "))
	if Np==None: Np = int(input("Input n_orb: "))
	with open(filename) as f:
		dim = f.readline()
		print("The dimension is "+dim)
		a = f.readlines()
	basis = [int(x) for x in a[::2]]
	basis_inv = []
	for thing in basis:
		basis_inv.append(np.sum([2**(Np-1-x) for x in range(len(bin(thing))-2) if bin(thing)[-x-1]=="1"]))
	coef  = a[1::2]
	toprint = dim+"".join(f"{x}\n{y}" for [x,y] in zip_longest(basis_inv,coef))
	with open(filename+"out","w+") as f:
		f.write(toprint)
	print("File saved successfully.")
	return

def read_decimal_basis_file(filename=None, basisformat = "dec", Np=None):
	if filename==None: filename=str(input("Input file name: "))
	with open(filename) as f:
		b = list(map(int,f.readlines()))
		if basisformat=="dec":
			return b
		elif basisformat=="index":
			#if Np==None: Np = int(input("Input n_orb: "))
			return list(map(dec_to_index,b))
			
def filter_zero(bnamein, vnamein, bnameout,vnameout):
	bas = []
	coe = []
	
	with open(bnamein) as f, open(vnamein) as g:
		for [x,y] in zip_longest(f.readlines(),g.readlines()):
			print(x[:-1],end="\t")
			print(y[:-1],end="\t")
			if abs(float(y))>=1e-14:
				bas.append(x)
				coe.append(y)
				print("recorded______________",end="\r",flush=1)
			else:
				print("removed________________________",end="\r",flush=1)
	
	f = open(bnameout,"w+")
	towrite = "".join(k for k in bas)
	f.write(towrite)
	f.close()
	
	f = open(vnameout,"w+")
	towrite = "".join(k for k in coe)
	f.write(towrite)
	f.close()
	return

def disk_coef(M,m):
	return np.sqrt(np.prod(np.arange(M)+1)*np.prod(np.arange(m)+1))

def sphere_coef(s,m):
	return np.prod(np.sqrt(np.arange(s-m)+1))/np.prod(np.sqrt(np.arange(s-m+1)+s+m+1))

def sphere_correction(s,x, north=True):
	if north:
		return sphere_coef(s+0.5,s+0.5-x-1)/sphere_coef(s,s-x)
	else:
		return sphere_coef(s+0.5,s+0.5-x)/sphere_coef(s,s-x)

def add_zero(dim,basis, coef,Np=None, basis_format="index",quiet = True, north=True):
	assert len(basis)==dim and len(coef)==dim
	if np.dot(coef,coef)<1e-10:
		if not quiet:
			print("Zero input vector -- Process terminated")
		return 0, 0
	if basis_format=="dec" or basis_format=="index":
		if Np==None: Np = int(input("Input n_orb: "))
	S = (Np-1)/2
	
	if basis_format=="dec":
		basis = list(map(dec_to_index, basis))
	
	if north:
		new_basis = [[x+1 for x in y] for y in basis]
	else:
		new_basis = [x for x in basis]           
	#new_coef  = np.array([coef[i]*np.prod([sphere_coef(S,S-x) for x in basis[i]])/np.prod([sphere_coef(S+0.5,S+0.5-x) for x in new_basis[i]]) for i in range(dim)])
	new_coef  = np.array([coef[i]*np.prod([sphere_correction(S,x,north) for x in basis[i]]) for i in range(dim)])
	norm = np.sqrt(np.dot(new_coef,new_coef))
	if not quiet:
		print(f"Check norm = {norm}")

	new_coef /= np.sqrt(np.dot(new_coef,new_coef))
	
	return new_basis, new_coef

def w2j(basis, coef, Np=None, basis_format="index",debug=False):
	if basis_format=="dec" or basis_format=="index":
		if Np==None: Np = int(input("Input n_orb: "))    
	if basis_format=="dec":
		basis = list(map(dec_to_index, basis))
	S = (Np-1)/2
	correction = np.array([np.prod([sphere_coef(S,x-S) for x in y]) for y in basis])
	if debug: print(correction)
	coef /= correction
	return basis, coef

def j2w(basis, coef, Np=None, basis_format="index", debug=False):
	if basis_format=="dec" or basis_format=="index":
		if Np==None: Np = int(input("Input n_orb: "))    
	if basis_format=="dec":
		basis = list(map(dec_to_index, basis))
	S = (Np-1)/2
	correction = np.array([np.prod([sphere_coef(S,x-S) for x in y]) for y in basis])
	if debug: print(correction)
	coef *= correction
	coef /= np.sqrt(np.dot(coef,coef))
	return basis, coef