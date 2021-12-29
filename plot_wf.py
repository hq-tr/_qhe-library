import sys
sys.path.append("/home/trung/_qhe-library/")
import sphere_wavefunctions as sphwf 
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm
#from celluloid import Camera

def plot_sphere_density(state, No, title=None, fname=None, invert=False, bosonic=False, ref=None):
	basis = state.get_basis_index(No)
	coef  = state.coef
	fig = plt.figure(figsize=(10,5))
	ax1 = fig.add_subplot(121, projection = "polar")
	ax  = fig.add_subplot(122, projection='3d')
	ax.view_init(5, 20)
	#ax.set_axis_off()
	#ax.set_xlabel("x")
	#ax.set_ylabel("y")
	#ax.set_zlabel("z")

	u = np.linspace(0, 2 * np.pi, 80)
	v = np.linspace(0, np.pi, 80)

	Theta, Phi = np.meshgrid(v, u)
	a = np.array([1/np.sqrt(2), -1/np.sqrt(2)])

	if invert:
		Density   = sphwf.get_density(basis,coef,No, np.pi-Theta, Phi, bosonic=bosonic)
	else:
		Density   = sphwf.get_density(basis,coef,No, Theta, Phi, bosonic = bosonic)


	q = ax1.pcolormesh(Phi, Theta,Density,vmin=np.min(Density),vmax=np.max(Density), shading = "gouraud")
	ax1.grid(True, color="k",linewidth=0.2)
	plt.colorbar(q)
	#ax1.set_xlabel(r"$\theta$")
	#ax1.set_ylabel(r"$\phi$")
	ax1.set_title(r"$\rho(\theta,\phi)$")
	# create the sphere surface
	#x=np.outer(np.cos(u), np.sin(v))
	#y=np.outer(np.sin(u), np.sin(v))
	#z=np.outer(np.ones(np.size(u)), np.cos(v))

	x = np.sin(Theta)*np.cos(Phi)
	y = np.sin(Theta)*np.sin(Phi)
	z = np.cos(Theta)

	#print(Theta.shape)
	#print(x.shape)

	#print(Density)

	print((np.min(Density), np.max(Density)))
	heatmap = cm.viridis(Density/np.max(Density))
	p = ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=heatmap)
	plt.tight_layout()
	if title != None:
		plt.suptitle(title)
	if fname==None:
		plt.savefig("density.pdf")
	else:
		plt.savefig(fname)
	"""
	Theta, Phi = np.meshgrid(theta, phi)
	print(Theta)
	print(Phi)

	a = np.array([1/np.sqrt(2), -1/np.sqrt(2)])

	Density   = sphwf.get_density([[1,4],[2,3]],a,5, Theta, Phi)

	plt.pcolormesh(Theta, Phi, Density)
	plt.colorbar()
	plt.savefig("density.pdf")
	"""
	return

def plot_sphere_density_animate(state, No, title=None, fname=None, invert=False, bosonic=False, ref=None):
	frame_rate = 2 # frame rates in fps
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=frame_rate, metadata=dict(artist='Me'), bitrate=1800)

	basis = state.get_basis_index(No)
	coef  = state.coef
	fig = plt.figure(figsize=(10,5))
	ax1 = fig.add_subplot(121, projection = "polar")
	ax  = fig.add_subplot(122, projection='3d')
	camera = Camera(fig)

	u = np.linspace(0, 2 * np.pi, 80)
	v = np.linspace(0, np.pi, 80)

	Theta, Phi = np.meshgrid(v, u)
	a = np.array([1/np.sqrt(2), -1/np.sqrt(2)])

	if invert:
		Density   = sphwf.get_density(basis,coef,No, np.pi-Theta, Phi, bosonic=bosonic)
	else:
		Density   = sphwf.get_density(basis,coef,No, Theta, Phi, bosonic = bosonic)


	q = ax1.pcolormesh(Phi, Theta,Density,vmin=np.min(Density),vmax=np.max(Density), shading = "gouraud")
	ax1.grid(True, color="k",linewidth=0.2)
	plt.colorbar(q)
	ax1.set_title(r"$\rho(\theta,\phi)$")

	x = np.sin(Theta)*np.cos(Phi)
	y = np.sin(Theta)*np.sin(Phi)
	z = np.cos(Theta)


	print((np.min(Density), np.max(Density)))
	heatmap = cm.viridis(Density/np.max(Density))
	p = ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=heatmap)
	plt.tight_layout()
	if title != None:
		plt.suptitle(title)

	rot_range = 2*np.pi*np.linspace(0,1)
	for rot in rot_range[:-1]:
		ax.view_init(5,rot)
		plt.draw()
		camera.snap()

	anim = camera.animate()
	anim.save("density.gif", writer=writer,dpi=300)
	#if fname==None:
	#	plt.savefig("density.pdf")
	#else:
	#	plt.savefig(fname)

	return

def plot_disk_density(state, No, title=None, fname=None, ref = None, bosonic=False, view = "polar"):
	assert view in ["polar", "3d"]
	basis = state.get_basis_index(No)
	coef  = state.coef
	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection=view)
	#ax1.axis("equal")
	#ax.set_xlabel("x")
	#ax.set_ylabel("y")
	#ax.set_zlabel("z")

	if view == "polar":
		Rmax = np.sqrt(2*No)
	else:
		Rmax = np.sqrt(2*No)-1

	r     = np.linspace(0, Rmax, 150)
	theta = np.linspace(0,2*np.pi, 150)

	R, T  = np.meshgrid(r,theta)


	X = R * np.cos(T)
	Y = R * np.sin(T)

	a = np.array([1/np.sqrt(2), -1/np.sqrt(2)])


	Density   = sphwf.get_density_disk(basis,coef, X, Y, bosonic=bosonic)

	if view == "polar":
		q = ax1.pcolormesh(T, R,Density,vmin=np.min(Density),vmax=np.max(Density), shading = "gouraud")
	else:
		q = ax1.plot_surface(X,Y,Density, cmap="viridis",linewidth=0)
	ax1.grid(True, color="k",linewidth=0.2)
	plt.colorbar(q, label=r"$\rho(x,y)$")
	plt.xlabel("x")
	plt.ylabel("y")
	#ax1.set_xlabel(r"$\theta$")
	#ax1.set_ylabel(r"$\phi$")
	#ax1.set_title(r"$\rho(z)$")
	# create the sphere surface
	#x=np.outer(np.cos(u), np.sin(v))
	#y=np.outer(np.sin(u), np.sin(v))
	#z=np.outer(np.ones(np.size(u)), np.cos(v))

	#x = np.sin(Theta)*np.cos(Phi)
	#y = np.sin(Theta)*np.sin(Phi)
	#z = np.cos(Theta)

	#print(Theta.shape)
	#print(x.shape)

	#print(Density)


	heatmap = cm.viridis(Density/np.max(Density))
	#p = ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=heatmap)

	if ref != None:
		if view == "polar":
			ref_ang = [np.angle(thing) for thing in ref]
			ref_amp = [abs(thing) for thing in ref]
			plt.plot(ref_ang, ref_amp, "ro")
		else:
			refx = [np.real(thing) for thing in ref]
			refy = [np.imag(thing) for thing in ref]
			plt.plot(refx, refy, "ro")

	#plt.tight_layout()
	if title != None:
		plt.suptitle(title)
	if fname==None:
		plt.savefig("density.pdf")
	else:
		plt.savefig(fname)
	"""
	Theta, Phi = np.meshgrid(theta, phi)
	print(Theta)
	print(Phi)

	a = np.array([1/np.sqrt(2), -1/np.sqrt(2)])

	Density   = sphwf.get_density([[1,4],[2,3]],a,5, Theta, Phi)

	plt.pcolormesh(Theta, Phi, Density)
	plt.colorbar()
	plt.savefig("density.pdf")
	"""
	return

if __name__ == "__main__":
	import FQH_states as FQH 
	#laughlin = FQH.fqh_state("test_state_Laughlin")
	laughlin = FQH.fqh_state((["000000000000001"], [1.0]))
	print(laughlin.format)
	#No = int(input("Input N_orb: "))
	No = 15
	plot_sphere_density(laughlin, No)
	#bas = ["001001001001", "001000110001", "010001001001"]
	#coef = np.array([0.2,0.6,0.3])
	#coef /= np.dot(coef,coef)
	#state = FQH.fqh_state((bas,coef))
	#plot_sphere_density(state,12)