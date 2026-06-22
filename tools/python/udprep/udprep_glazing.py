import numpy as np

def calc_TR_phi(T_0, R_0, phi, d):
    # Compute transmittance and reflectance at angle of incidence phi
    # for solar radiation (average wavelength 0.898 microns)
    #
    # Inputs:
    #   T_0       - normal-incidence transmittance T(0)
    #   R_0       - normal-incidence reflectance  R(0)
    #   phi       - angle of incidence in rad
    #   d        - glass thickness in meters
    #
    # Outputs:
    #   T_phi    - transmittance at angle phi
    #   R_phi    - reflectance  at angle phi

    lambda_ = 0.898e-6  # solar average wavelength (m)

    # Solve for interface reflectivity at normal incidence
    beta = T_0 ** 2 - R_0 ** 2 + 2 * R_0 + 1
    rho0 = (beta - np.sqrt(beta ** 2 - 4 * (2 - R_0) * R_0)) / (2 * (2 - R_0))

    # Index of refraction
    n = (1 + np.sqrt(rho0)) / (1 - np.sqrt(rho0))

    # Extinction coefficient and absorption coefficient
    kappa = -(lambda_ / (4 * np.pi * d)) * np.log((R_0 - rho0) / (rho0 * T_0))
    alpha = 4 * np.pi * kappa / lambda_

    # Refraction angle via Snell's law
    phi_prime = np.arcsin(np.sin(phi) / n)

    # Fresnel reflectivity at phi (unpolarized)
    rho_phi = 0.5 * (
        ((n * np.cos(phi) - np.cos(phi_prime)) / (n * np.cos(phi) + np.cos(phi_prime))) ** 2
        + ((n * np.cos(phi_prime) - np.cos(phi)) / (n * np.cos(phi_prime) + np.cos(phi))) ** 2
    )

    #  Interface transmissivity at phi
    tau_phi = 1 - rho_phi

    # Transmittance and reflectance at phi
    expterm = np.exp(-alpha * d / np.cos(phi_prime))
    T_phi = (tau_phi ** 2 * expterm) / (1 - rho_phi ** 2 * expterm ** 2)
    R_phi = rho_phi * (1 + T_phi * expterm)

    return T_phi, R_phi


def calc_TRcoated_phi(T_0, R_0, phi):
    # Compute transmittance and reflectance of coated glass at angle phi
    # using regression fit based on uncoated reference glass curves
    #
    # Inputs:
    #   T_0       - normal-incidence transmittance T(0)
    #   R_0       - normal-incidence reflectance  R(0)
    #   phi       - angle of incidence in rad
    # Outputs:
    #   T_phi    - transmittance at angle phi
    #   R_phi    - reflectance  at angle phi

    # Polynomial coefficients
    tau_clr = np.array([-0.0015, 3.355, -3.840, 1.460, 0.0288])
    rho_clr = np.array([0.999, -0.563, 2.043, -2.532, 1.054])

    tau_bnz = np.array([-0.002, 2.813, -2.341, -0.05725, 0.599])
    rho_bnz = np.array([0.997, -1.868, 6.513, -7.862, 3.225])

    # Evaluate polynomials at phi
    c = np.cos(phi)
    cpow = np.array([1, c, c ** 2, c ** 3, c ** 4])  # [cos^0, cos^1, cos^2, cos^3, cos^4]

    tau_clr_phi = np.dot(tau_clr, cpow)
    rho_clr_phi = np.dot(rho_clr, cpow) - tau_clr_phi

    tau_bnz_phi = np.dot(tau_bnz, cpow)
    rho_bnz_phi = np.dot(rho_bnz, cpow) - tau_bnz_phi

    # Apply to coated glass based on T(0)
    if T_0 > 0.645:
        T_phi = T_0 * tau_clr_phi
        R_phi = R_0 * (1 - rho_clr_phi) + rho_clr_phi
    else:
        T_phi = T_0 * tau_bnz_phi
        R_phi = R_0 * (1 - rho_bnz_phi) + rho_bnz_phi

    return T_phi, R_phi


def calc_TRA_EP(T, Rf, Rb): # recursive calculation of transmittance, reflectance and absorptance of the glazing system
    
    N = len(T)
    # optical properties of subsystems
    Tw_f = np.zeros(N)  
    Tw_b = np.zeros(N)
    Rfw_f = np.zeros(N)
    Rfw_b = np.zeros(N)
    Rbw_f = np.zeros(N)
    Rbw_b = np.zeros(N)
    Aw = np.zeros(N)

    # Forward T R (forward means when radiation comes from the outside)
    Tw_f[0] = T[0]
    Rfw_f[0] = Rf[0]
    Rbw_f[0] = Rb[0]

    for i in range(1, N):
        T_f = np.zeros(2)
        R_f = np.zeros(2)
        R_b = np.zeros(2)
        T_f[0] = Tw_f[i - 1]
        R_f[0] = Rfw_f[i - 1]
        R_b[0] = Rbw_f[i - 1]
        T_f[1] = T[i]
        R_f[1] = Rf[i]
        R_b[1] = Rb[i]
        Tw_f[i] = T_f[0] * T_f[1] / (1 - R_f[1] * R_b[0])
        Rfw_f[i] = R_f[0] + T_f[0] ** 2 * R_f[1] / (1 - R_f[1] * R_b[0])
        Rbw_f[i] = R_f[1] + T_f[1] ** 2 * R_b[0] / (1 - R_b[0] * R_f[1])

    # Backward T R
    Tw_b[N - 1] = T[-1]
    Rfw_b[N - 1] = Rf[-1]
    Rbw_b[N - 1] = Rb[-1]

    for i in range(N - 2, -1, -1):
        T_f = np.zeros(2)
        R_f = np.zeros(2)
        R_b = np.zeros(2)
        T_f[1] = Tw_b[i + 1]
        R_f[1] = Rfw_b[i + 1]
        R_b[1] = Rbw_b[i + 1]
        T_f[0] = T[i]
        R_f[0] = Rf[i]
        R_b[0] = Rb[i]
        Tw_b[i] = T_f[0] * T_f[1] / (1 - R_f[1] * R_b[0])
        Rfw_b[i] = R_f[0] + T_f[0] ** 2 * R_f[1] / (1 - R_f[1] * R_b[0])
        Rbw_b[i] = R_f[1] + T_f[1] ** 2 * R_b[0] / (1 - R_b[0] * R_f[1])

    # A
    # 1st layer
    Aw[0] = (1 - T[0] - Rf[0]) + T[0] * Rfw_b[1] * (1 - T[0] - Rb[0]) / (1 - Rfw_b[1] * Rbw_f[0])

    for i in range(1, N - 1):
        Aw[i] = (
            Tw_f[i - 1] * (1 - T[i] - Rf[i]) / (1 - Rfw_b[i] * Rbw_f[i - 1])
            + Tw_f[i] * Rfw_b[i + 1] * (1 - T[i] - Rb[i]) / (1 - Rfw_b[i] * Rbw_f[i - 1])
        )
    # for i=2:N-1
    #     Aw_f(i)=Tw_f(i-1)*(1-T(i)-Rf(i))/(1-Rfw_b(i)*Rbw_f(i-1))...
    #            + Tw_f(i)*Rfw_b(i+1)*(1-T(i)-Rb(i))/(1-Rfw_b(i+1)*Rbw_f(i));

    # Nst layer
    Aw[N - 1] = Tw_f[N - 2] * (1 - T[N - 1] - Rf[N - 1]) / (1 - Rfw_b[N - 1] * Rbw_f[N - 2])

    Tw = Tw_f[-1]
    Rfw = Rfw_f[-1]
    Rbw = Rbw_f[-1]

    return Tw, Rfw, Rbw, Aw


def calc_optiproperties(T_0, Rf_0, Rb_0, d_g, d_gas, phi):
    # parameters initialization
    N = len(T_0)  # N glazing layers
    T_phi = np.zeros(N)  # transmittance at an incident angle for each glazing layer
    Rf_phi = np.zeros(N)  # front reflectance at an incident angle for each glazing layer
    Rb_phi = np.zeros(N)  # back reflectance at an incident angle for each glazing layer
    T_D = np.zeros(N)  # hemispherical transmittance for each glazing layer
    Rf_D = np.zeros(N)  # hemispherical front reflectance for each glazing layer
    Rb_D = np.zeros(N)  # hemispherical back reflectance for each glazing layer
    d = np.sum(d_g) + np.sum(d_gas)  # thickness of the entire glazing system [m]
    
    # optical properties for direct radiation
    # calculate optical properties for each glass in a specific incident angle
    for j in range(N):
        if Rf_0[j] != Rb_0[j]:
            T_phi[j], Rf_phi[j] = calc_TRcoated_phi(T_0[j], Rf_0[j], phi)
            _, Rb_phi[j] = calc_TRcoated_phi(T_0[j], Rb_0[j], phi)
        else:
            T_phi[j], Rf_phi[j] = calc_TR_phi(T_0[j], Rf_0[j], phi, d_g[j])
            _, Rb_phi[j] = calc_TR_phi(T_0[j], Rb_0[j], phi, d_g[j])

    # calculate optical properties for the entire system
    if N == 1:
        Tw = T_phi
        Rfw = Rf_phi
        Rbw = Rb_phi
        Aw = 1 - Tw - Rfw
    else:
        Tw, Rfw, Rbw, Aw = calc_TRA_EP(T_phi, Rf_phi, Rb_phi)

    # optical properties for diffuse radiaiton
    deg = np.arange(0, 91, 1)
    T = np.zeros(len(deg))
    Rf = np.zeros(len(deg))
    Rb = np.zeros(len(deg))
    phi = np.deg2rad(deg)

    for i in range(N):
        for j in range(len(deg)):
            if Rf_0[i] != Rb_0[i]:
                T[j], Rf[j] = calc_TRcoated_phi(T_0[i], Rf_0[i], phi[j])
                _, Rb[j] = calc_TRcoated_phi(T_0[i], Rb_0[i], phi[j])
            else:
                T[j], Rf[j] = calc_TR_phi(T_0[i], Rf_0[i], phi[j], d_g[i])
                _, Rb[j] = calc_TR_phi(T_0[i], Rb_0[i], phi[j], d_g[i])
        weight = 2 * np.cos(phi) * np.sin(phi)
        T_D[i] = np.trapezoid(T * weight, phi)
        Rf_D[i] = np.trapezoid(Rf * weight, phi)
        Rb_D[i] = np.trapezoid(Rb * weight, phi)

    if N == 1:
        TwD = T_D
        RfwD = Rf_D
        RbwD = Rb_D
        AwD = 1 - TwD - RfwD
    else:
        TwD, RfwD, RbwD, AwD = calc_TRA_EP(T_D, Rf_D, Rb_D)

    return Tw, Rfw, Rbw, Aw, TwD, RfwD, RbwD, AwD