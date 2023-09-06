'''
Analyzes the electric field of a Uniform distributed charge using multiple methods:
    1) FFT
    2) Analytical integration

The computed electric field from the FFT method is then compared with the results from the analytical method.
Additionally, the code visualizes the distribution of the charge particles.

version: draft02_1
'''

import numpy as np
import PyPIC.FFT_OpenBoundary as PIC_FFT
import PyPIC.geom_impact_ellip as ell
from scipy.constants import e, epsilon_0
import matplotlib.pyplot as plt


# ###################### Constants and parameters #########################
R_cham = 1e-1  # Chamber radius [m]
R_charge = 2e-2 # [m] Scaling parameter equivalent to the radius of the uniformly distributed charge in the 01_uniform_fft_analytic.py file
Dh = 1e-3  # Grid spacing [m]
N_part_gen = 1000000 # Number of generated particles
electron_charge = e
eps0 = epsilon_0 

# Define chamber geometry
chamber = ell.ellip_cham_geom_object(x_aper=R_cham, y_aper=R_cham)


# ##################### Particle generation ###############################
theta = 2 * np.pi * np.random.random(N_part_gen)        
r = np.sqrt(np.random.random(N_part_gen)) * R_charge    
x_part = r * np.cos(theta)                 
y_part = r * np.sin(theta)
mask_keep = x_part**2 + y_part**2 < R_cham**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = np.ones_like(x_part)
x_probes = np.linspace(-R_cham, R_cham, 1000)
y_probes = np.zeros_like(x_probes)


# ################## Analytical (using simple Gauss's law)  ###############
E_r_thx = [np.sum(x_part**2 + y_part**2 < x**2) * electron_charge / eps0 / (2 * np.pi * x) for x in x_probes]


# ############################### FFT ####################################
picFFT = PIC_FFT.FFT_OpenBoundary(x_aper=chamber.x_aper, y_aper=chamber.y_aper, dx=Dh/2, dy=Dh, fftlib='pyfftw')
picFFT.scatter(x_part, y_part, nel_part, charge=e)
picFFT.solve()
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)


# #################### Visualization #######################
fig = plt.figure(figsize=(8, 4))

# First plot (Electric field)
ax0 = plt.subplot(1, 2, 1)
ax0.plot(x_probes, E_r_thx, label="Analytic(Gauss's law)", color='green')
ax0.plot(x_probes, Ex_FFT, label='FFT-open boundary', color='red', linestyle='--')
ax0.set_xlim([-R_cham-0.01, R_cham+0.01])
ax0.legend(loc='lower right', fontsize=9)
ax0.set_ylabel('Ex on the x axis [V/m]')
ax0.set_xlabel('x [m]')
ax0.set_title("Uniform distribution at y=0:\nFFT , Analytic(Gauss's law)")

# Second plot (Chamber boundary and particle distribution)
ax1 = plt.subplot(1, 2, 2)
ax1.scatter(x_part, y_part, s=0.5, label='charged particles')
ax1.set_xlim([-R_cham-0.01, R_cham+0.01])
ax1.set_ylim([-R_cham-0.01, R_cham+0.01])
ax1.set_title('Particle distribution')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

plt.tight_layout()
#plt.savefig('Uniform_FFT_analytic.png')
plt.show()
