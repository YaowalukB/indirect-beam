# Indirect-beam

## Description:
This repository contains all the work completed during the summer school on the study of charged particle distribution. The primary focus was on the perpendicular side relative to the beam's travel direction.

## Dependencies:
- **PyPIC:** The main package used for various particle-in-cell operations.
  - Modules: FiniteDifferences_ShortleyWeller_SquareGrid, FFT_OpenBoundary, geom_impact_ellip, and more.
  
- **numpy:** Utilized for numerical operations.

- **scipy:** Required for constants and various other functionalities.

- **matplotlib:** Deployed for plotting and visualization.

## Project steps & associated code:

0. **Initialization and setup**
   - Description: Setting up the necessary environment, libraries, and tools.

1. **Electric field analysis of uniform disk charge distribution**
   - Approach: FFT open-boundary and analytical.
   - Specifics:
     - The charge distribution is modeled as a uniform disk.
     - A circular geometry is utilized for the chamber and the charge distribution.
   - [view code](./src/01_uniform_fft_analytic.py)
   - [View result](./results/01_1.png)

2. **Electric field analysis of Gaussian charge distribution**
   - Approach: FFT open-boundary and analytical.
   - Specifics:
     - Gaussian distribution had a standard deviation half of the uniform disk charge.
     - The Gaussian distribution in this step utilizes the same standard deviation (sigma) for both x and y axes, making it symmetric.
   - [view code](./src/02_gaussian_fft_analytic.py)
   - [View result](./results/02_1.png)

3. **Semi-analytic approach on Gaussian charge distribution**
   - Approach: Baseeti-Erskien formula was applied.
   - Specifics: Gaussian distribution with a standard deviation half of the uniform disk charge.
   - [view code](./src/03_gaussian_semi_analytic.py)
   - [View result](./results/03_1.png)
   - **Notes:**
     - In process `03`, an elliptical geometry was introduced for both the chamber and the charge distribution.
     - Introduced `shape_factor` for elliptical shaping of the charge distribution and chamber.
     - Due to the introduction of the elliptical geometry, parameters like chamber radius and charge radius are now defined distinctly for the x and y axes.
     - This process requires `scipy.special` for the `wofz` function used in the Bassetti-Erskine method.


4. **Electric field study using FDSW on uniform charge distribution**
   - Approach: Shortley-Weller finite difference method (FDSW).
   - [view code](./src/04_uniform_fdsw.py)
   - [View result](./results/04_1.png)
   - **Notes:**
     - While process `03` introduced an elliptical geometry, in process `04`, we chose to revert back to a spherical model.
     - The decision to transition back was influenced by some challenges and unanticipated results associated with the elliptical model in process `03`.
     - The spherical geometry, defined by `shape factor` set to 1, was employed for both the chamber and the charge distribution. The framework remains adaptable for future explorations into elliptical models.
     - We recognize the significance of the elliptical model and believe that it has potential for more detailed investigation. However, for the immediate goals of this study and in light of the time constraints, the spherical model was deemed more suitable.
     - Introduced `y_offset` to adjust the vertical position of the charge distribution's origin, aiming to gain insights into the boundary effects on the field.

5. **Electric field study using FDSW on Gaussian charge distribution**
   - Approach: Shortley-Weller finite difference method (FDSW).
   - Specifics: Gaussian distribution with a standard deviation half of the uniform disk charge.
   - Other setup are the same as 4.
   - [view code](./src/05_gaussian_fdsw.py)
   - [View result](./results/05_1.png)

6. **Electric field from force of round beams on single test particle**
   - Approach: Simplifying assumption of round beams.
   - Specifics: Applied to Gaussian charge distribution with standard deviation half of the uniform disk charge.
   - [view code](./src/06_gaussian_round_beam_force.py)
   - [View result](./results/06_1.png)
