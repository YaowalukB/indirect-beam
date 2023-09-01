'''
Analyzes the electric field of a Gaussian distributed charge using multiple methods:
    1) FFT
    2) Analytical integration
    3) Bassetti-Erskine formula

The computed electric field from the FFT method is then compared with the results from the analytical methods.
Additionally, the code visualizes the distribution of the charge particles.

version: draft03_1
'''

import numpy as np
import PyPIC.FFT_OpenBoundary as PIC_FFT
import PyPIC.geom_impact_ellip as ell
from scipy.constants import e, epsilon_0
import matplotlib.pyplot as plt
from scipy.special import wofz


# ###################### Constants and parameters #########################
shape_factor = 1.5 # Defines the elliptical shape
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
mean = 0  # Assuming centered at origin
sigma = R_charge_x / 2  #standard deviation of the gaussian distribution

# For particles generation and calculation, the sigma(standard deviation) is categorized into three types:
sigma_y = sigma
sigma_x = shape_factor * sigma_y
sigma_u = sigma_x

x_part = np.random.normal(mean, sigma_x, N_part_gen)
y_part = np.random.normal(mean, sigma_y, N_part_gen)

mask_keep = (x_part**2 / R_cham_x**2) + (y_part**2 / R_cham_y**2) < 1
x_part = x_part[mask_keep]
y_part = y_part[mask_keep]

nel_part = np.ones_like(x_part)
x_probes = np.linspace(-R_cham_x, R_cham_x, 1000)
y_probes = np.zeros_like(x_probes)
charge_density = len(x_part) * electron_charge / 1 # Linear charge density per 1 meter


# ################## Analytic integral ###############
def electric_field_gaussian(x, y):
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

# linearization of the analytic integral near the origin
x_near_org = np.linspace(-R_charge_x/4, R_charge_x/4, 10000)
y_near_org = np.zeros_like(x_near_org)
E_near_origin =  (-charge_density / ((2 * np.pi * eps0) * (sigma_u * (sigma_x + sigma_y)))) * x_near_org


# ############################### BASSETTI-ERSKEIN ####################################
def electric_field_bassetti_erskine(x, y):
    z1 = (x + 1j * y) / (np.sqrt(2 * (sigma_x**2 - sigma_y**2)))
    z2 = (x * (sigma_y / sigma_x) + 1j * y * (sigma_x / sigma_y)) / (np.sqrt(2 * (sigma_x**2 - sigma_y**2)))
    w_z1 = wofz(z1)
    w_z2 = wofz(z2)
    prefactor = (charge_density / (2 * eps0 * np.sqrt(2 * np.pi * (sigma_x**2 - sigma_y**2))))
    # Calculate the electric fields in x and y directions.
    result = (-prefactor * (w_z1 - (np.exp(-x**2 / (2 * sigma_x**2) - y**2 / (2 * sigma_y**2)) * w_z2)))
    E_x = result.imag
    E_y = result.real
    #print('z1,z2', z1, z2)
    return E_x, E_y

# Compute the electric fields using the Bassetti-Erskine formula.
E_x_B = []
E_y_B = []
for x, y in zip(x_probes, y_probes):
    Ex, Ey = electric_field_bassetti_erskine(x, y)
    E_x_B.append(Ex)
    E_y_B.append(Ey)


# ############################### FFT ####################################
picFFT = PIC_FFT.FFT_OpenBoundary(x_aper=chamber.x_aper, y_aper=chamber.y_aper, dx=Dh/2, dy=Dh, fftlib='pyfftw')
picFFT.scatter(x_part, y_part, nel_part)
picFFT.solve()
Ex_FFT, Ey_FFT = picFFT.gather(x_probes, y_probes)


# #################### Visualization #######################
fig = plt.figure(figsize=(8, 6))

# First plot (Electric field)
ax0 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
ax0.plot(x_probes, Ex_FFT, label='FFT', color='purple')
ax0.plot(x_probes, Ex_integral, label = 'Analytical integral', color='green')
ax0.plot(x_near_org, E_near_origin, label='Linearlization near origin', color='black')
ax0.plot(x_probes, E_x_B, label='Bassetti_Erskine Ex', color='red')
ax0.set_xlim([-R_cham_x-0.01, R_cham_x+0.01])
ax0.legend(loc='upper right')
ax0.set_ylabel('Ex on the x axis [V/m]')
ax0.set_title('FFT vs analytic: Gaussian charge')

# Second plot (Chamber boundary and particle distribution)
ax1 = plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
ax1.scatter(x_part, y_part, s=0.5, label='charged particles')
ax1.set_xlim([-R_cham_x-0.01, R_cham_x+0.01])
ax1.set_ylim([-R_cham_y-0.01, R_cham_y+0.01])
ax1.set_title('Particle distribution')
ax1.legend(loc='upper right')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

plt.tight_layout()
plt.show()
