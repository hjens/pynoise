import numpy as np
import pylab as pl
import pynoise as pn

redsh = 9.533
time = 1000

par = pn.Parameters()
par.set_nu_c(pn.z_to_nu(redsh))

tsys = lambda nu: 140. + 60*(nu/300.)**(-2.55)
aeff = lambda nu: 526.*(nu/150.)**(-2.)


par.set_t(time)
par.set_num_tel(48)
par.set_uv_range(2.*860)
par.set_fov(5.)
par.set_num_pol(2)

#par.set_uv_grid_from_telescopes('/home/hjens/Powerpaper/lofar_coordinates.dat', ha_range=[-12,12])

r_core = 150.
r_out = 1500.
par.set_uv_range(r_out*2.)
r_distr = lambda r: (r>10)*(r < r_core)+(r < r_out)*(r>r_core)*(r/r_core)**(-2.)
par.set_uv_grid_from_function(r_distr)

par.set_nu_range_cubic()

par.set_uv_taper(lambda r: r < 500)

par.set_tsys(tsys)
par.set_aeff(aeff)
#par.set_tsys(140. + 60*(par.get_nu_c()/300.)**(-2.55))
#par.set_aeff(526.*(par.get_nu_c()/150.)**(-2.))

par.print_params()

image  = par.get_image_slice()


pl.figure()
pn.plot_image_slice(par, image)

pl.figure()
pn.plot_uv_coverage_radial(par)

pl.show()
