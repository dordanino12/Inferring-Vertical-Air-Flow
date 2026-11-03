"""Cloud microphysics -> extinction coefficient (Sec. 3.4, Eqs. 1-5 of the paper)."""
import numpy as np

K_B = 1.380649e-23        # Boltzmann constant [J/K]
M_AIR = 4.81e-26          # mass of an air molecule [kg]
RHO_WATER = 1e6           # water density [g/m^3]


def process_cloud_vars(QN, NC, T, p):
    """
    QN : cloud water mixing ratio [g/kg], shape [Z, Y, X]
    NC : droplet number concentration [cm^-3], shape [Z, Y, X]
    T  : absolute temperature [K], shape [Z, Y, X]
    p  : pressure profile [mbar], shape [Z]
    Returns LWC [g/m^3], Reff [um], beta_ext [1/m] (each [Z, Y, X]).
    """
    Z, X, Y = QN.shape
    NC = NC * 1e6                                                   # [m^-3]
    P = np.multiply(p * 100, np.ones([X, Y, Z])).transpose((2, 0, 1))  # [Pa]
    rho_air = (P * M_AIR) / (K_B * T)                               # Eq. 1 [kg/m^3]
    LWC = QN * rho_air                                              # Eq. 2 [g/m^3]
    droplet_vol = (LWC / (NC + 1e-15)) / RHO_WATER                  # Eq. 3 [m^3]
    reff_m = ((3 * droplet_vol) / (4 * np.pi)) ** (1 / 3)           # Eq. 4 [m]
    # Eq. 5 with Q_ext = 2:  beta = Q_ext*pi*N*Reff^2 = 3*LWC / (2*rho_w*Reff)
    beta_ext = (3 * LWC) / (2 * RHO_WATER * reff_m + 1e-15)         # [1/m]
    return LWC, reff_m * 1e6, beta_ext
