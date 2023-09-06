'''
Analyzes the electric field of a Gaussian distributed charge using multiple methods:
    1) FFT
    2) Analytical integration
    3) FDSW: Finite Differences with Shortley-Weller boundary conditions
    4) Electric field from force of round beams on single test particle

The computed electric field from the FFT method is then compared with the results from the analytical methods.
Additionally, the code visualizes the distribution of the charge particles.

version: draft06_1
'''

import numpy as np
import PyPIC.FFT_OpenBoundary as PIC_FFT
import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as PIC_FDSW
import PyPIC.geom_impact_ellip as ell
from scipy.constants import e, epsilon_0
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ###################### Constants and parameters #########################
shape_factor = 1 # Defines the elliptical shape
R_cham_y = 1e-1  # Chamber radius on y axis [m]
R_cham_x = shape_factor * R_cham_y # Chamber radius on x axis [m]
R_charge_y = 2e-2 # [m] Scaling parameter equivalent to the radius of the uniformly distributed charge in the 01_uniform_fft_analytic.py file
R_charge_x = shape_factor * R_charge_y
Dh = 1e-3  # Grid spacing [m]
N_part_gen = 1000000 # Number of generated particles
electron_charge = e
eps0 = epsilon_0 

# Define chamber geometry
chamber = ell.ellip_cham_geom_object(x_aper=R_cham_x, y_aper=R_cham_y)


# ##################### Particle generation ###############################
mean_x = 0  # Assuming centered at origin in x-direction
sigma = R_charge_x / 2  #standard deviation of the gaussian distribution
y_offset = 0.04

# Adjusting mean for y-direction by y_offset
mean_y = y_offset

# For particles generation and calculation, the sigma(standard deviation) is categorized into three types:
sigma_y = sigma
sigma_x = shape_factor * sigma_y
sigma_u = sigma_x

x_part = np.random.normal(mean_x, sigma_x, N_part_gen)
y_part = np.random.normal(mean_y, sigma_y, N_part_gen)

mask_keep = (x_part**2 / R_cham_x**2) + (y_part**2 / R_cham_y**2) < 1
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]
nel_part = np.ones_like(x_part)
charge_density = len(x_part) * electron_charge / 1 # Linear charge density per 1 meter

x_probes = np.linspace(-R_cham_x, R_cham_x, 1000)
y_probes = np.full(1000, y_offset)
    

# ############################### FFT ####################################
picFFT = PIC_FFT.FFT_OpenBoundary(x_aper=chamber.x_aper, y_aper=chamber.y_aper, dx=Dh/2, dy=Dh, fftlib='pyfftw')
picFFT.scatter(x_part, y_part, nel_part, charge=e)
picFFT.solve()
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)


# ####################### FD Shortley-Weller #############################
picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb=chamber, Dh=Dh)
picFDSW.scatter(x_part, y_part, nel_part, charge=e)
picFDSW.solve()
Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)


# #################### Visualization #######################
fig = plt.figure(figsize=(8, 4))
boundary_x = R_cham_x

# First plot (Electric field)
ax0 = plt.subplot(1, 2, 1)  # 2 rows, 1 column, first subplot
ax0.axvline(boundary_x, color='grey', linestyle=(0, (1, 1)), label='Boundary')
ax0.axvline(-boundary_x, color='grey', linestyle=(0, (1, 1)))
ax0.plot(x_probes, Ex_FDSW, label='FD-closed boundary', color='blue')
ax0.plot(x_probes, Ex_FFT, label='FFT-open boundary', color='red', linestyle='--')
ax0.set_xlim([-R_cham_x-0.01, R_cham_x+0.01])
ax0.legend(loc='lower right', fontsize=9)
ax0.set_ylabel('Ex [V/m]')
ax0.set_xlabel('x [m]')
ax0.set_title('Gaussian distribution at y=0\nopen vs. closed boundary approaches')

# Second plot (Chamber boundary and particle distribution)
ax1 = plt.subplot(1, 2, 2)  # 2 rows, 1 column, second subplot
ellipse = patches.Ellipse((0,-y_offset), 2*R_cham_x, 2*R_cham_y, fill=False, color='black', linestyle='-', label=f"Vacuum pipe boundary\nat y_offset=-{y_offset}")
ax1.add_patch(ellipse)
ax1.axhline(0, color='grey', linestyle=(0, (3, 1)), label='y = 0 (Field computation line)')
ax1.scatter(x_part, y_part-y_offset, s=0.5, label='Charged particles')
ax1.set_xlim([-R_cham_x-0.01, R_cham_x+0.01])
ax1.set_ylim([-R_cham_y-0.01-y_offset, R_cham_y+0.01-y_offset])
ax1.set_title('Particle distribution')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

plt.tight_layout()
#plt.savefig('Gaussian_y0_OpenVsClosed.png')
#plt.savefig('Gaussian_yoffset_OpenVsClosed.png')
plt.show()
