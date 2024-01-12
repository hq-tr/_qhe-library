# Updated 2021-07-01
import numpy as np 
from itertools import accumulate
import misc
import plot_wf as plw 
import angular_momentum as am

def factorial(x):
	return np.prod(np.arange(x)+1)

def sqfactorial(x):
	return np.prod(np.sqrt(np.arange(x)+1))

def is_binary_array_quick(string_array): # Check if all elements in array is binary (only containing "0" or "1")
	return is_binary_string(string_array[0]) and is_binary_string(string_array[1])

def is_binary_array(string_array):
	return all([is_binary_string(x) for x in string_array])

def is_binary_string(input_string):
	content = set(input_string)
	if content == {"0","1"} or content == {"0"} or content == {"1"}:
		return True
	else:
		return False

class fqh_state:
	def __init__(self, param=None, quiet=False):
		if param == None:
			self.basis = []
			self.coef  = np.array([], dtype=float)
			self.format = "null"
		elif type(param) is str: # -------- If input is file
			try:
				file = param
				with open(file) as f:
					dim = f.readline()
					a = f.readlines()
					basis = [x[:-1] for x in a[::2]]
					if is_binary_array(basis):  # Check if file is in binary
						self.basis = basis
						self.format = "binary"
					elif " " in basis[0]:		# Check if file is bosonic binary
						if not quiet: print("Please use FQH_states.fqh_state_boson() for bosonic states.")
						self.basis = []
						self.coef  = np.array([])
					else:						# If file is decimal
						self.basis = list(map(int, basis))
						self.format = "decimal"
					if "j" in a[1]:
						self.coef  = np.array(list(map(complex, a[1::2])))
					else:
						self.coef  = np.array(list(map(float, a[1::2])))
			except FileNotFoundError:
				if not quiet: print(f"{file}: File not found")
				self.basis = []
				self.coef  = np.array([], dtype=float)
				self.format = "null"
		else:					# --------- If input is tuple of (basis, coef)
			try:
				if len(param) == 1:
					if type(param[0]) is str:
						try:
							with open(param[0]) as f:
								self.basis = list(map(int, f.readlines()))
							self.coef = np.zeros(len(basis))
							self.format = "decimal"
						except Exception as err:
							if not quiet: 
								print("To initialize a zero state with given basis file, please ensure the basis is in decimal format.")
								print(err)
					elif type(param[0]) is list:
						basis = param[0]
						if np.issubdtype(type(basis[0]), np.integer): # If basis is decimal
							self.basis = list(map(int, basis))
							self.format = "decimal"
						elif is_binary_array(basis):	# If basis is binary
							self.basis = basis
							self.format = "binary"
						self.coef = np.zeros(len(basis))


				elif type(param[0]) is str and type(param[1]) is str:
					try:
						with open(param[0]) as f:
							self.basis = list(map(int, f.readlines()))

						with open(param[1]) as f:
							self.coef = list(map(float, f.readlines()))
						assert len(self.basis) == len(self.coef), "Dimension mismatch"
						self.format = "decimal"
					except Exception as err:
						if not quiet:
							print("To initialize state with separate basis and coefficient files, please ensure the basis is in decimal format and coefficients are real.")
							print(err)

				elif len(param[0]) == len(param[1]):
					basis = param[0]
					if np.issubdtype(type(basis[0]), np.integer): # If basis is decimal
						self.basis = list(map(int, basis))
						self.format = "decimal"
					elif is_binary_array(basis):	# If basis is binary
						self.basis = basis
						self.format = "binary"
					elif " " in basis[0]:			# If basis is binary but boson
						if not quiet: print("Please use FQH_states.fqh_state_boson() for bosonic states.")
						self.basis = []
						self.coef  = np.array([])
					else:							# If basis is decimal but in strings
						self.basis = list(map(int, basis))
						self.format = "decimal"
					if any(np.iscomplex(param[1])):
						self.coef  = np.array(param[1])
					else:
						self.coef  = np.array(param[1])
				else:
					if not quiet: print("Dimension mismatch")
					self.basis = []
					self.coef  = np.array([],dtype=float)
					self.format = "null"
			except Exception as err:
				if not quiet:
					print("Please initialize state with either wavefunction file or a tuple of basis and coefficients")
					print(f"ERROR: {err}")
				self.basis = []
				self.coef  = np.array([],dtype=float)
				self.format = "null"

	def __add__(self, other):
		if self.format == other.format or self.format == "null" or other.format == "null":
			basis, coef1, coef2 = misc.collate_vectors(self.basis, self.coef, other.basis, other.coef,get_array=True)
			coef = coef1 + coef2
			return fqh_state((basis,coef))
		else: 
			print("Basis format mismatch")
			return fqh_state()

	def __mul__(self, other):
		#assert type(other) is float or type(other) is int or type(other) is complex, f"* is used for scalar multiplication. Use self.dot() for overlap.\ntype(other)={type(other)}"
		return fqh_state((self.basis, self.coef*other))

	def __rmul__(self, other):
		#assert type(other) is float or type(other) is int or type(other) is complex, f"* is used for scalar multiplication. Use self.dot() for overlap. \ntype(other)={type(other)}"
		return fqh_state((self.basis, self.coef*other))

	def __sub__(self,other):
		return self + (-1)*other

	def dim(self):
		return len(self.basis)

	def normalize(self, tol = 1e-10):
		norm = np.dot(np.conj(self.coef), self.coef)
		if norm < tol:
			print("Warning: Normalizing a null state")
			print(f"norm = {norm}")
		self.coef /= np.sqrt(norm)
		return

	def norm(self):
		return np.dot(np.conj(self.coef), self.coef)

	def printwf(self, filename=None):
		dim = len(self.basis)
		t = f"{dim}\n"+"".join(f"{self.basis[i]}\n{self.coef[i]:.20f}\n" for i in range(dim))
		if filename==None:
			print(t)
		else:
			with open(filename, "w+") as f:
				f.write(t)
		return 

	def sphere_normalize(self, No = None):
		if self.format == "decimal":
			if No == None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index(x) for x in self.basis]
		S = (No-1)/2
		multiplier  = np.array([np.prod([No*S*misc.sphere_coef(S,S-x) for x in y]) for y in basis_index])
		self.coef *= multiplier
		#print(multiplier)
		self.normalize()
		#print(f"check norm = {self.norm()}")
		return

	def disk_normalize(self, No = None):
		if self.format == "decimal":
			if No == None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index(x) for x in self.basis]
		multiplier = np.array([np.prod([sqfactorial(x)/(np.sqrt(2**x)*sqfactorial(No//3)) for x in y]) for y in basis_index])
		#print(basis_index)
		with open("multiplier","w+") as f:
			f.write("\n".join(str(x) for x in multiplier))
		self.coef *= multiplier 
		self.normalize()
		return

	def sphere_norm(self,No=None):
		if self.format == "decimal":
			if No == None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index(x) for x in self.basis]
		S = (No-1)/2
		multiplier  = sqfactorial(No)*np.array([np.prod([misc.sphere_coef(S,S-x) for x in y]) for y in basis_index])
		coef = self.coef*multiplier
		return np.dot(np.conj(coef), coef)

	def disk_norm(self, No = None):
		if self.format == "decimal":
			if No == None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index(x) for x in self.basis]
		multiplier = np.array([np.prod([sqfactorial(x)/(np.sqrt(2**x)*sqfactorial(No//3)) for x in y]) for y in basis_index])
		#print(basis_index)
		#with open("multiplier","w+") as f:
		#	f.write("\n".join(str(x) for x in multiplier))
		coef = self.coef * multiplier 
		return np.dot(np.conj(coef), coef)


	def overlap(self, other):
		if self.format == other.format:
			basis, coef1, coef2 = misc.collate_vectors(self.basis, self.coef, other.basis, other.coef,get_array=True)
			#print(f"norm 1 = {np.dot(coef1,coef1)}")
			#print(f"norm 2 = {np.dot(coef2,coef2)}")
			return np.dot(np.conj(coef1),coef2)
		elif self.format == "null" or other.format == "null":
			return 0
		else: 
			print("Basis format mismatch")
			return 0

	def perp(self, other):
		ovl = self.overlap(other)
		ret = self - ovl * other
		ret.normalize()
		return ret

	def format_convert(self, No=None,invert=False):
		if self.format=="decimal":
			if No == None: No = int(input("Input N_orb: "))
			basis = [misc.decimal_to_binary(x,No,invert= invert) for x in self.basis]
			self.basis = basis
			self.format = "binary"
		elif self.format=="binary":
			No = len(self.basis[0])
			basis = [misc.binary_to_decimal(x,invert=invert) for x in self.basis]
			self.basis = basis
			self.format = "decimal"

		return

	def get_basis_index(self,No=None):
		if self.format == "decimal":
			if No==None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index(x) for x in self.basis]
		return basis_index 

	def invert(self,No=None):
		if self.format == "binary":
			basis_new = [x[::-1] for x in self.basis]
			self.basis = basis_new
			return
		else:
			if No == None: No = int(input("Input N_orb: "))
			self.format_convert(No,invert=False)
			self.format_convert(invert=True)
			return

	def plot_disk_density(self, fname=None, No=None, ref = None, title=None):
		if self.format == "decimal":
			if No==None: No = int(input("Input N_orb: "))
		elif self.format == "binary":
			No = len(self.basis[0])
		plw.plot_disk_density(self, No, fname=fname, ref=ref, title=title)
		return

	def plot_sphere_density(self, fname=None, No=None, ref = None, title=None):
		if self.format == "decimal":
			if No==None: No = int(input("Input N_orb: "))
		elif self.format == "binary":
			No = len(self.basis[0])
		plw.plot_sphere_density(self, No, fname=fname, ref=ref, title=title)
		return

	def s2d(self, No=None):
		if self.format == "decimal":
			if No == None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index(x) for x in self.basis]
		S = (No-1)/2
		multiplier = np.array([np.prod([sqfactorial(x)/np.sqrt(2**x) for x in y]) for y in basis_index])
		divider    = sqfactorial(No)*np.array([np.prod([misc.sphere_coef(S,S-x) for x in y]) for y in basis_index])

		self.coef *= multiplier/divider
		self.normalize()
		return

	def d2s(self, No=None):
		if self.format == "decimal":
			if No == None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index(x) for x in self.basis]
		S = (No-1)/2
		divider    = np.array([np.prod([sqfactorial(x)/np.sqrt(2**x) for x in y]) for y in basis_index])
		multiplier = sqfactorial(No)*np.array([np.prod([misc.sphere_coef(S,S-x) for x in y]) for y in basis_index])

		self.coef *= multiplier/divider
		self.normalize()
		return

	def Lplus(self, n=1):
		if self.format == "decimal":
			No = int(input("Input N_orb: "))
			basis = self.basis
		elif self.format == "binary":
			No = len(self.basis[0])
			basis = [misc.binary_to_decimal(x) for x in self.basis]
		coef = self.coef
		for i in range(n):
			basis, coef = am.Lplus(basis, coef, No)

		if len(basis)==0:
			print("State annihilated")
			return fqh_state()
		else:
			state = fqh_state((basis, coef))
			if self.format == "binary":
				state.format_convert(No)

			return state


	def Lminus(self, n=1):
		if self.format == "decimal":
			No = int(input("Input N_orb: "))
			basis = self.basis
		elif self.format == "binary":
			No = len(self.basis[0])
			basis = [misc.binary_to_decimal(x) for x in self.basis]
		coef = self.coef
		for i in range(n):
			basis, coef = am.Lminus(basis, coef, No)


		if len(basis)==0:
			print("State annihilated")
			return fqh_state()
		else:
			state = fqh_state((basis, coef))
			if self.format == "binary":
				state.format_convert(No)

			return state

class fqh_state_boson:
	def __init__(self, param=None):
		if param == None:
			self.basis = []
			self.coef  = np.array([], dtype=float)
			self.format = "null"
		elif type(param) is str: # -------- If input is file
			try:
				file = param
				with open(file) as f:
					dim = f.readline()
					a = f.readlines()
					basis = [x[:-1] for x in a[::2]]
					if " " in basis[0]:		# Check if file is binary
						self.basis = basis
						self.format = "binary"
					elif is_binary_array(basis):  # Check if file is in binary but fermion
						print("Please use FQH_states.fqh_state() for fermionic states.")
						self.basis = []
						self.coef  = np.array([])
					else:						# If file is decimal
						self.basis = list(map(int, basis))
						self.format = "decimal"
					if a[1][0] == "(":
						self.coef  = np.array(list(map(complex, a[1::2])))
					else:
						self.coef  = np.array(list(map(float, a[1::2])))
			except FileNotFoundError:
				print(f"{file}: File not found")
				self.basis = []
				self.coef  = np.array([], dtype=float)
				self.format = "null"
		else:					# --------- If input is tuple of (basis, coef)
			try:
				if len(param) == 1:  	# ---------- If only basis is specified
					if type(param[0]) is str:
						try:
							with open(param[0]) as f:
								basis = list(map(str, f.readlines()))
								assert " " in basis[0]
								self.basis = [x.strip("\n") for x in basis]
								self.coef  = np.zeros(len(basis))
								self.format = "binary"
						except Exception as err:
							print("To initialize a bosonic zero state with given basis file, please ensure the basis is in binary format.")
							print(err)
					elif type(param[0]) is list:	
						basis = param[0]
						if np.issubdtype(type(basis[0]), np.integer): # If basis is decimal
							self.basis = list(map(int, basis))
							self.format = "decimal"
						elif " " in basis[0]:			# If basis is binary
							self.basis = basis
							self.format = "binary"
						elif is_binary_array(basis):	# If basis is binary but fermion
							print("Please use FQH_states.fqh_state() for fermionic states.")
							self.basis = []
							self.coef  = np.array([])
						else:							# If basis is decimal but in strings
							self.basis = list(map(int, basis))
							self.format = "decimal"
				# ----- If both basis and coefficients are specified
				elif type(param[0]) is str and type(param[1]) is str: 
					try:
						with open(param[0]) as f:
							basis = list(map(str, f.readlines()))
							assert " " in basis[0]
							self.basis = [x.strip("\n") for x in basis]
						with open(param[1]) as f:
							self.coef = list(map(float, f.readlines()))
						assert len(self.basis) == len(self.coef), "Dimension mismatch"
						self.format = "binary"
					except Exception as err:
						print("To initialize state with separate basis and coefficient files, please ensure the basis is in binary format and coefficients are real.")
						print(err)
						self.basis = []
						self.coef  = np.array([])
						self.format = "null"
				elif len(param[0]) == len(param[1]):
					basis = param[0]
					if np.issubdtype(type(basis[0]), np.integer): # If basis is decimal
						self.basis = list(map(int, basis))
						self.format = "decimal"
					elif " " in basis[0]:			# If basis is binary
						self.basis = basis
						self.format = "binary"
					elif is_binary_array(basis):	# If basis is binary but fermion
						print("Please use FQH_states.fqh_state() for fermionic states.")
						self.basis = []
						self.coef  = np.array([])
					else:							# If basis is decimal but in strings
						self.basis = list(map(int, basis))
						self.format = "decimal"
					if any(np.iscomplex(param[1])):
						self.coef  = np.array(param[1])
					else:
						self.coef  = np.array(param[1])
				else:
					self.basis = []
					self.coef  = np.array([],dtype=float)
					self.format = "null"
			except Exception as err:
				print("Please initialize state with either wavefunction file or a tuple of basis and coefficients")
				print(f"ERROR: {err}")
				self.basis = []
				self.coef  = np.array([],dtype=float)
				self.format = "null"

	def __add__(self, other):
		if self.format == other.format or self.format == "null" or other.format == "null":
			basis, coef1, coef2 = misc.collate_vectors(self.basis, self.coef, other.basis, other.coef,get_array=True)
			coef = coef1 + coef2
			return fqh_state_boson((basis,coef))
		else: 
			print("Basis format mismatch")
			return fqh_state_boson()

	def __mul__(self, other):
		assert type(other) is float or type(other) is int or type(other) is complex, "* is used for scalar multiplication. Use self.dot() for overlap"
		return fqh_stat_boson((self.basis, self.coef*other))

	def __rmul__(self, other):
		assert type(other) is float or type(other) is int or type(other) is complex, "* is used for scalar multiplication. Use self.dot() for overlap"
		return fqh_state_boson((self.basis, self.coef*other))

	def __sub__(self,other):
		return self + (-1)*other

	def dim(self):
		return len(self.basis)

	def normalize(self, tol = 1e-10):
		norm = np.dot(np.conj(self.coef), self.coef)
		#print(norm)
		if norm < tol:
			print("Warning: Normalizing a null state")
		self.coef /= np.sqrt(norm)
		return

	def norm(self):
		return np.dot(np.conj(self.coef), self.coef)

	def printwf(self, filename=None):
		dim = len(self.basis)
		t = f"{dim}\n"+"".join(f"{self.basis[i]}\n{self.coef[i]:.20f}\n" for i in range(dim))
		if filename==None:
			print(t)
		else:
			with open(filename, "w+") as f:
				f.write(t)
		return 

	def format_convert(self, No=None,invert=False):
		if self.format=="decimal":
			print("decimal-to-binary conversion for bosonic state isn't implemented yet lol")
			#if No == None: No = int(input("Input N_orb: "))
			#basis = [misc.decimal_to_binary(x,No,invert= invert) for x in self.basis]
			#self.basis = basis
			#self.format = "binary"
		elif self.format=="binary":
			No = len(self.basis[0])
			basis = [misc.binary_to_decimal_boson(x,invert=invert) for x in self.basis]
			self.basis = basis
			self.format = "decimal"

		return

	def invert(self,No=None):
		if self.format == "binary":
			basis_new = [x[::-1] for x in self.basis]
			self.basis = basis_new
			return
		else:
			"Inversion only works for binary format in the current version."
			return
			
	def overlap(self, other):
		if self.format == other.format:
			basis, coef1, coef2 = misc.collate_vectors(self.basis, self.coef, other.basis, other.coef,get_array=True)
			#print(f"norm 1 = {np.dot(coef1,coef1)}")
			#print(f"norm 2 = {np.dot(coef2,coef2)}")
			return np.dot(np.conj(coef1),coef2)
		elif self.format == "null" or other.format == "null":
			return 0
		else: 
			print("Basis format mismatch")
			return 0

	def get_basis_index(self,No=None):
		if self.format == "decimal":
			if No==None: No = int(input("Input N_orb: "))
			basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index_boson(x) for x in self.basis]
		return basis_index

	def sphere_normalize(self, No = None):
		if self.format == "decimal":
			print("The routine doesn't work for basis in decimal format yet")
			return
			#No = int(input("Input N_orb: "))
			#basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index_boson(x) for x in self.basis]
		S = (No-1)/2
		multiplier  = np.array([np.prod([sqfactorial(No)*misc.sphere_coef(S,S-x) for x in y]) for y in basis_index])
		multiplier2 = np.array([np.sqrt(2) if "2" in x else 1 for x in self.basis])
		self.coef *= multiplier
		#print(self.coef)
		self.normalize()
		#print(f"check norm = {self.norm()}")
		return

	def disk_normalize(self, No = None):
		if self.format == "decimal":
			print("The routine doesn't work for basis in decimal format yet")
			return
			#No = int(input("Input N_orb: "))
			#basis_index = [misc.dec_to_index(x) for x in self.basis]
		elif self.format == "binary":
			No = len(self.basis[0])
			basis_index = [misc.binary_to_index_boson(x) for x in self.basis]
		multiplier = np.array([np.prod([sqfactorial(x)/np.sqrt(2**x) for x in y]) for y in basis_index])
		multiplier2 = np.array([np.sqrt(2) if "2" in x else 1 for x in self.basis])
		#print(basis_index)
		#with open("multiplier","w+") as f:
		#	f.write("\n".join(str(x) for x in multiplier))
		self.coef *= multiplier
		self.normalize()
		return

	def plot_disk_density(self, fname=None, No=None,ref=None, title=None):
		if self.format == "decimal":
			if No==None: No = int(input("Input N_orb: "))
		elif self.format == "binary":
			No = (len(self.basis[0])+1)/2
		plw.plot_disk_density(self, No, title=title,ref=ref, fname=fname, bosonic=True)
		return

	def plot_sphere_density(self, fname=None, No=None, ref = None, title=None):
		if self.format == "decimal":
			if No==None: No = int(input("Input N_orb: "))
		elif self.format == "binary":
			No = len(self.basis[0].replace(" ", ""))
		plw.plot_sphere_density(self, No, fname=fname, ref=ref, title=title, bosonic=True)
		return

	def Lplus(self, n=1):
		if self.format == "decimal":
			No = int(input("Input N_orb: "))
			basis = self.basis
		elif self.format == "binary":
			No = len(self.basis[0])
			basis = [misc.binary_to_decimal(x) for x in self.basis]
		coef = self.coef
		for i in range(n):
			basis, coef = am.Lplus(basis, coef, No)

		if len(basis)==0:
			print("State annihilated")
			return fqh_state()
		else:
			state = fqh_state((basis, coef))
			if self.format == "binary":
				state.format_convert(No)

			return state


	def Lminus(self, n=1):
		if self.format == "decimal":
			No = int(input("Input N_orb: "))
			basis = self.basis
		elif self.format == "binary":
			No = len(self.basis[0])//2 + 1
			basis = [misc.binary_to_decimal(x) for x in self.basis]
		coef = self.coef
		for i in range(n):
			basis, coef = am.Lminus(basis, coef, No)


		if len(basis)==0:
			print("State annihilated")
			return fqh_state()
		else:
			state = fqh_state((basis, coef))
			if self.format == "binary":
				state.format_convert(No)

			return state

def com(n,k):
	if k<=n//2:
		return np.prod(np.arange(k)+(n-k+1))/np.prod(np.arange(k)+1)
	else:
		return com(n,n-k)

def sqcom(n,k):
	if k<=n//2:
		return np.prod(np.sqrt((np.arange(k)+n-k+1)/(np.arange(k)+1)))
	else:
		return sqcom(n,n-k)

## Two-quasihole state 
class two_qh_state:
	def __init__(self, N_e = 2, N_o = 4):
		self.z_0 = 0
		self.N_e = N_e
		self.N_o = N_o 

	def get_pos(self):
		return self.z_0

	def get_bosonic_coef(self):
		factor = np.array([sqfactorial(x) for x in np.arange(self.N_e+1)])
		#factor = 1 / np.array([sqcom(self.N_e+1, x) for x in range(self.N_e+1)])
		if self.z_0 < 4:
			return (self.z_0/np.sqrt(2))**np.arange(self.N_e+1)/(factor*self.N_o)
		else:
			return (self.z_0/np.sqrt(2))**(-self.N_e+np.arange(self.N_e+1))/factor*(np.sqrt(self.N_o)**self.N_e)

	def get_bosonic_coef_sphere(self, d_phi=False):
		factor = np.array([(sqfactorial(self.N_e-x))/(sqfactorial(x)) for x in np.arange(self.N_e+1)])
		m = np.arange(self.N_e+1)
		if self.z_0 < 4:
			return (self.z_0**m)*factor/self.N_o
		else:
			return (self.z_0**(-self.N_e+m))*factor*self.N_o

	def get_bosonic_polynomial_coef(self):
		#print("getting factors")
		factor = np.array([factorial(x) for x in np.arange(self.N_e+1)])
		#print(factor)
		if self.z_0 < 4:
			return np.array([np.prod([self.N_o*self.z_0/y for y in range(1,x+1)]) for x in range(self.N_e+1)])
		else:
			return np.array([np.prod([self.N_o/(self.z_0*y) for y in range(1,x+1)]) for x in range(self.N_e+1)])

	def get_fermionic_polynomial_coef(self):
		m = np.arange(self.N_e+1)
		if self.z_0 < 4:
			return (-self.z_0)**m/self.N_o
		else:
			return (-self.z_0)**(-self.N_e+m)*self.N_o

## Useful functions

def prune_sort(state):
	# remove all basis with zero coefficients and sort the basis in decreasing coefficients
	dim = len(state.basis)

	collect = {state.coef[i]: state.basis[i] for i in range(dim)}

	coef = list(compress(state.coef, state.coef))
	coef.sort(reverse=True)
	basis = [collect[x] for x in coef]

	new = FQH.fqh_state((basis, coef))
	return new

def orthonormalize(states, tol = 1e-12):
	basis = [x.basis for x in states]
	coef  = [x.coef  for x in states]
	all_basis, all_coef = misc.collate_many_vectors(basis, coef)
	##print(all_coef)
	o_vec = misc.orthonormalize(all_coef, tol=tol)
	return [fqh_state((all_basis, x)) for x in o_vec]


__all__ = ["fqh_state", "fqh_state_boson", "orthonormalize", "two_qh_state", "prune_sort"]

fqh_state.__doc__ = """
An  fqh_state can be initialized by two methods:
(1) a string containing the name of the text file that stores the wavefunction, or
(2) a tuple (basis, coef) containing the basis vectors and the corresponding coefficients.

fqh_state variables come with an in-built vector space structure. 
"""
if __name__=="__main__":
	a = fqh_state_boson((["2 0 0 0"], [1.]))
	b = a.Lminus()
	b.printwf()