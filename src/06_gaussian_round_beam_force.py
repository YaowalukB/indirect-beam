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
R_charge_y = 4e-2 # [m] Scaling parameter equivalent to the radius of the uniformly distributed charge in the 01_uniform_fft_analytic.py file
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
y_offset = 0.00

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

# ################## Analytic integral ###############
def electric_field_gaussian(x, y):
    y = y - y_offset  # introduce the offset
    t_lower = 0
    t_upper = 10
    
    def integrand(t):
        return (np.exp(-x**2 / (2 * sigma_x**2 + t) - y**2 / (2 * sigma_y**2 + t))) / ((sigma_u**2 + t) * np.sqrt((sigma_x**2 + t) * (sigma_y**2 + t)))
    t = np.linspace(t_lower,t_upper,100000)
    integral_result = np.trapz(integrand(t), t)
    return -charge_density / (4 * np.pi * eps0) * np.array([x, y]) * integral_result

# Compute the electric fields using analytical integration
Ex_integral = []
Ey_integral = []
for x, y in zip(x_probes, y_probes):
    Ex, Ey = electric_field_gaussian(x, y)
    Ex_integral.append(Ex)
    Ey_integral.append(Ey)


# ############################ Analytic eq.12(round beam) ##################
def electric_field_roundbeam(x, y):
   y = y - y_offset  
   n = charge_density
   r2 = x**2 + y**2
   e = electron_charge
   B = 0
   sigma = sigma_x
   return ((-n * e**2 * (1 + B**2) / (2 * np.pi * eps0)) * (np.array([x, y]) / r2) * (1 - np.exp(-r2 / (2 * sigma**2))))/e**2

Ex_eq12 = []
Ey_eq12 = []
for x, y in zip(x_probes, y_probes):
    Ex, Ey = electric_field_roundbeam(x, y)
    Ex_eq12.append(Ex)
    Ey_eq12.append(Ey)
    

# ############################### FFT ####################################
picFFT = PIC_FFT.FFT_OpenBoundary(x_aper=chamber.x_aper, y_aper=chamber.y_aper, dx=Dh/2, dy=Dh, fftlib='pyfftw')
picFFT.scatter(x_part, y_part, nel_part)
picFFT.solve()
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)


# ####################### FD Shortley-Weller #############################
picFDSW = PIC_FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb=chamber, Dh=Dh)
picFDSW.scatter(x_part, y_part, nel_part)
picFDSW.solve()
Ex_FDSW, Ey_FDSW = picFDSW.gather(x_probes, y_probes)


# #################### Visualization #######################
fig = plt.figure(figsize=(8, 6))
boundary_x = R_cham_x

# First plot (Electric field)
ax0 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
ax0.axvline(boundary_x, color='red', linestyle='--', label='Boundary')
ax0.axvline(-boundary_x, color='red', linestyle='--')
ax0.plot(x_probes, Ex_FFT, label='FFT', color='purple')
ax0.plot(x_probes, Ex_FDSW, label='FD ShortleyWeller', color='blue')
ax0.plot(x_probes, Ex_integral, label = 'Analytical integral', color='green')
ax0.plot(x_probes, Ex_eq12, label='Analytic eq.12(round beam)', color = 'pink')
ax0.set_xlim([-R_cham_x-0.01, R_cham_x+0.01])
ax0.legend(loc='upper right')
ax0.set_ylabel('Ex on the x axis [V/m]')
ax0.set_title('FFT vs analytic: Gaussian charge')

# Second plot (Chamber boundary and particle distribution)
ax1 = plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
ellipse = patches.Ellipse((0,0), 2*R_cham_x, 2*R_cham_y, fill=False, color='black', linestyle='-', label="Chamber boundary")
ax1.add_patch(ellipse)
ax1.axhline(y_offset, color='black', linestyle='--', label=f"y_offset= {y_offset}")
ax1.scatter(x_part, y_part, s=0.5, label='charged particles')
ax1.set_xlim([-R_cham_x-0.01, R_cham_x+0.01])
ax1.set_ylim([-R_cham_y-0.01, R_cham_y+0.01])
ax1.set_title('Particle distribution')
ax1.legend(loc='upper right')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

plt.tight_layout()
plt.show()
