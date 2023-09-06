'''
Analyzes the electric field of a Gaussian distributed charge using multiple methods:
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


# ##################### Gaussian Particle generation ###############################
mean = 0  # Assuming centered at origin
sigma = R_charge / 2  #standard deviation of the gaussian distribution
x_part = np.random.normal(mean, sigma, N_part_gen)
y_part = np.random.normal(mean, sigma, N_part_gen)

mask_keep = x_part**2 + y_part**2 < R_cham**2
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = np.ones_like(x_part)
x_probes = np.linspace(-R_cham, R_cham, 1000)
y_probes = np.zeros_like(x_probes)
charge_density = len(x_part) * electron_charge / 1 # Linear charge density per 1 meter

# ##################### Uniform Particle generation ###############################
theta = 2 * np.pi * np.random.random(N_part_gen)        
r = np.sqrt(np.random.random(N_part_gen)) * R_charge    
x_parts = r * np.cos(theta)                 
y_parts = r * np.sin(theta)
mask_keep = x_parts**2 + y_parts**2 < R_cham**2
x_parts = x_parts[mask_keep]
y_parts = y_parts[mask_keep]

# ################## Analytical (using simple Gauss's law)  ###############
E_r_thx = [np.sum(x_parts**2 + y_parts**2 < x**2) * electron_charge / eps0 / (2 * np.pi * x) for x in x_probes]


# ############################ Analytic eq.12(round beam) ##################
def electric_field_roundbeam(x, y): 
   n = charge_density
   r2 = x**2 + y**2
   e = electron_charge
   return ((n / (2 * np.pi * eps0)) * (np.array([x, y]) / r2) * (1 - np.exp(-r2 / (2 * sigma**2))))

Ex_eq12 = []
Ey_eq12 = []
for x, y in zip(x_probes, y_probes):
    Ex, Ey = electric_field_roundbeam(x, y)
    Ex_eq12.append(Ex)
    Ey_eq12.append(Ey)

# linearization of the analytic integral near the origin
x_near_org = np.linspace(-R_charge/2, R_charge/2, 10000)
y_near_org = np.zeros_like(x_near_org)
E_near_origin =  charge_density * x_near_org / ((2 * np.pi * eps0) * (sigma * (sigma + sigma)))


# ############################### FFT ####################################
picFFT = PIC_FFT.FFT_OpenBoundary(x_aper=chamber.x_aper, y_aper=chamber.y_aper, dx=Dh/2, dy=Dh, fftlib='pyfftw')
picFFT.scatter(x_part, y_part, nel_part, charge=e)
picFFT.solve()
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)


# #################### Visualization #######################
fig = plt.figure(figsize=(8, 4))

# First plot (Electric field)
ax0 = plt.subplot(1, 2, 1)
ax0.plot(x_near_org, E_near_origin, label='Linearization', color='black')
ax0.plot(x_probes, Ex_eq12, label='analytic (round beam)', color = 'blue')
ax0.plot(x_probes, Ex_FFT, label='FFT-open boundary', color='red', linestyle='--')
ax0.plot(x_probes, E_r_thx, label="Analytic uniform distribution", color='lightgreen', linestyle='--')
ax0.set_xlim([-R_cham-0.01, R_cham+0.01])
ax0.legend(loc='lower right', fontsize=9)
ax0.set_ylabel('Ex on the x axis [V/m]')
ax0.set_xlabel('x [m]')
ax0.set_title('Gaussian distribution at y=0:\nFFT , Round beam, linearization')

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
plt.savefig('Gaussian_FFT_analytic_linearization.png')
plt.show()
