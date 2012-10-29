import numpy as np
import parameters as p

def get_visibility_noise(Tsys, epsilon, Aeff, dnu, nu_c, T, n_tel, uvgrid, num_pol = 1, seed=None):
	''' Calculate a noise realization in visibility space.

	Parameters:
		* Tsys --- System temperature in K
		* epsilon --- Antenna efficiency (0 < epsilon < 1)
		* Aeff --- Effective area in m^2
		* dnu --- Frequency resolution in MHz
		* nu_c --- Central frequency in MHz
		* T --- Observing time in hours
		* n_tel --- Number of telescopes
		* uvgrid --- uv grid
	kwargs:
		* num_pol = 1 --- number of polarizations
		* seed = None -- random number seed, if None use random seed

	Returns:
		complex array of dimensions NxM, containing real and imaginary noise in mK'''

	#Parameters needed for noise calculation
	C = 1.96/epsilon

	#The total integration time for all visibilites 
	num_baselines = n_tel*(n_tel-1)/2.
	total_time = T*3600.*num_baselines 

	#RMS noise 
	dV = C*(Tsys/500.)*(500./Aeff)*(dnu**(-0.5)) #Jy

	#Convert from Jy to mK
	dBdT = 6.9e2*(nu_c/150.)**2 #mJy-> mK
	dV *= 1.e3/dBdT
	if num_pol == 2:
		dV /= np.sqrt(2.)

	#Generate random noise
	if seed:
		np.random.seed(seed)
	sqrtuv = np.sqrt(uvgrid*total_time)
	noise = np.zeros(uvgrid.shape,dtype=np.complex)
	Vre = dV*np.random.normal(0., 1., uvgrid.shape)/sqrtuv
	Vim = dV*np.random.normal(0., 1., uvgrid.shape)/sqrtuv
	noise = Vre + 1j*Vim

	#Fill NaNs with zeros
	noise[np.where(noise!=noise)] = 0.

	return noise

