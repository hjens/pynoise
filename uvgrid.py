#This file contains methods for calculating the uv-coverage of a 
#telescope array.
#The methods here are not primarily intended to be used separately, but will
#be called from the Parameters structure

import numpy as np
import parameters as p


def get_uv_grid_from_function(rho_uv, fov, uv_range = 2000.):
	'''
	Calculate uv coverage grid from a function of r specifying the baseline distribution
	as a function of baseline length

	Parameters:
		* rho_uv --- callable taking one parameter. The function should give the baseline
		density as a function of baseline length.
		* fov --- field of view in degrees

	kwargs:
		* uv_range = 2000 --- u_max-u_min in wavelengths

	Returns:
		* uv grid, normalized so that the integral is 1
	'''

	#Determine grid size
	delta_u = 180./(fov*np.pi)
	gridn = round(float(uv_range)/delta_u)

	#Make an array with distance to center
	y,x = np.indices((gridn, gridn))
	center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	r = np.hypot(x-center[0], y-center[1])

	#Avoid numerical problems at the center
	r[center[0],center[1]] = r[center[0]+1,center[1]]

	#Scale to wavelength units
	r *= float(uv_range)/float(gridn)

	#Apply the function
	uv_grid = rho_uv(r)

	#Normalize
	norm_grid = uv_grid.astype('float')/float(uv_grid.sum())

	return norm_grid

def get_uv_grid_from_telescopes(tel_positions, fov, nu_c, ha_range, decl = 90, ha_step = 50, uv_range = 2000, mirror_points=False):
	'''
	Calculate the uv coverage of a telescope array.
		* tel_positions --- array with telescope x,y,z positions in m, cartesian geocentric coordinates
		Must have the shape (N,3) where N is the number of telescopes. 
		* fov --- the field of view in degrees
		* nu_c --- central frequency in MHz
		* ha_range --- tuple with start and stop hour angle (in hours)

	kwargs:
		* ha_step = 50 --- time resolution in uv calculation in seconds
		* decl = 90 --- declination of the source, in degrees
		* uv_range = 2000 --- u_max-u_min in wavelengths. This parameter specified the cutoff for the grid. The actual covered range is set by the telescope positions.
		* mirror_points = False --- whether to include (-u,-v) points

	Returns:
		* uv grid, normalized so that the integral i 1
	'''


	#Convert to radians
	d = np.radians(decl)
	ha_step = ha_step/3600.
	hour_angles = np.arange(ha_range[0], ha_range[1], ha_step)
	h = hour_angles*np.pi/12.

	#Update number of telescopes for parameters
	#params.set_num_tel(tel_positions.shape[0])

	#Calculate baslines
	baselines = get_all_baselines(tel_positions)

	uvw = np.array([[], [], []])

	from numpy import sin,cos

	print 'Calculating uvw coverage...'
	#Construct uvw matrix for all hour angles
	for hn in h:
		proj_mtrx = np.array([
			[sin(hn), 			cos(hn),			0],
			[-sin(d)*cos(hn),	sin(hn)*sin(d),		cos(d)],
			[cos(d)*cos(hn),	-cos(d)*sin(hn), 	sin(d)]  
			])
		uvwn = np.dot(proj_mtrx, baselines)
		uvw = np.hstack([uvw, uvwn])
		#Add mirror points
		if mirror_points:
			uvw = np.hstack([uvw, -uvwn]) 

	wavel = 300./nu_c
	uvw /= wavel

	#Grid uv points
	uv = uvw[:2,:].transpose()
	gridmax = uv_range/2.
	delta_u = 180./(fov*np.pi)
	gridn = int(round((uv_range/delta_u)))
	uedges = np.linspace(-gridmax,gridmax,gridn+1)
	vedges = np.linspace(-gridmax,gridmax,gridn+1)

	uv_grid, bins = np.histogramdd(uv, bins=[uedges,vedges])

	return uv_grid.astype('float')/float(uv_grid.sum())#, bins

def get_rho_uv_from_antenna_distribution(rho_ant, fov, uv_range, wavel, num_points_phi=301, num_points_r=1001):
	''' 
	Calculate uv coverage grid from a function of r giving the 
	antenna density as a function of distance from the array center.
	Still somewhat experimental. Please check the results for numerical
	problems.

	Parameters:
		* rho_ant --- callable taking one parameter. This function should give the 
		density of antennae in the array as a function of distance (in meters) 
		from the array center
		* fov --- field of view in degrees
		* uv_range --- u_max-u_min in wavelengths
		* wavel --- the wavelength in meters
	kwargs:
		* num_points_phi = 301 --- number of sample points for phi when integrating
		* num_points_r = 10001 --- number of sample points for r when integrating

	Returns:
		* callable, uv density as a function of u
	'''

	from scipy.integrate import simps
	from scipy.interpolate import interp1d

	intf = lambda r,phi,u: r*rho_ant(r)*rho_ant(np.sqrt(r**2+u**2*wavel**2-2.*r*wavel*u*np.cos(phi)))

	phi_points = np.linspace(0,2.*np.pi, num_points_phi)
	r_points = np.linspace(0.1,uv_range*3, num_points_r)
	

	#Integrate over r
	def func_r(u, phi):
		fvals = intf(r_points, phi, u)
		return simps(fvals, r_points)

	#Integrate over phi
	def rho_u(u):
		fvals = np.array([func_r(u, phi) for phi in phi_points])
		return 2.*np.pi*simps(fvals, phi_points)
	
	#Calculate it
	uvals = np.linspace(0,uv_range/2., 100.)
	rho_u = np.array([rho_u(u) for u in uvals])
	func = interp1d(uvals, rho_u, kind='linear', bounds_error = False, fill_value=0)

	#Make a grid
	#grid = get_uv_grid_from_function(func, fov, uv_range)

	return func


def get_all_baselines(tel_positions, unique_only = False):
	'''
	Return all the possible baselines from the telescope configuration
	Return in form [[x1, x2, ..., xn], [y1, y2, ..., yn]] 
	For internal use.
	'''

	print 'Calculating baselines...'

	tel_x = tel_positions[:,0]
	tel_y = tel_positions[:,1]
	tel_z = tel_positions[:,2]
	baselines = np.array([[], [], []])

	#Calculate all the baselines
	for i in range(len(tel_x)):
		#base_x = np.abs(tel_x[i+1:]-tel_x[i])
		base_x =tel_x[i+1:]-tel_x[i]
		base_y = tel_y[i+1:]-tel_y[i]
		base_z = tel_z[i+1:]-tel_z[i]
		baselines_temp = np.vstack([base_x, base_y, base_z])
		baselines = np.hstack([baselines, baselines_temp])

	#Filter out all the non-unique baselines
	if unique_only:
		baselines = baselines.transpose()
		d = {}
		for b in baselines:
			t = tuple(b)
			d[t] = 0#d.get(t,0)+1
		ret = np.array(d.keys()).transpose()
	else:
		ret = baselines

	return ret

def get_tapered_uv_grid(uvgrid, uv_range, taper_func):
	''' 
	Multiply a uv grid with a taper function

	Parameters:
		* uvgrid --- the uv grid, an NxN array
		* uv_range --- u_max - u_min 
		* taper_func --- callable. A function of one variable - the 
			baseline length in wavelengths - which will be multiplied
			with the uv grid

	'''

	#Determine grid size
	gridn = uvgrid.shape[0]

	#Make an array with distance to center
	y,x = np.indices((gridn, gridn))
	center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
	r = np.hypot(x-center[0], y-center[1])

	#Avoid numerical problems at the center
	r[center[0],center[1]] = r[center[0]+1,center[1]]

	#Scale to wavelength units
	r *= float(uv_range)/float(gridn)

	#Multiply with function and return
	return uvgrid*taper_func(r)
