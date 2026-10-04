import numpy as np

# coated glass with a normal-incidence transmittance above this value uses the
# clear reference curve in calc_layer_TR_coated, otherwise the bronze one
T_0_CLEAR_MIN = 0.645


def calc_layer_TR_uncoated(T_0, R_0, phi, d_g):
    # Compute transmittance and reflectance at angle of incidence phi
    # for solar radiation (average wavelength 0.898 microns)
    #
    # Inputs:
    #   T_0       - normal-incidence transmittance T(0)
    #   R_0       - normal-incidence reflectance  R(0)
    #   phi       - angle of incidence in rad
    #   d_g       - glass thickness in meters
    #
    # Outputs:
    #   T_phi    - transmittance at angle phi
    #   R_phi    - reflectance  at angle phi

    wavelength = 0.898e-6  # solar average wavelength (m)

    # Solve for interface (Fresnel) reflectivity at normal incidence
    beta = T_0 ** 2 - R_0 ** 2 + 2 * R_0 + 1
    R_fresnel_0 = (beta - np.sqrt(beta ** 2 - 4 * (2 - R_0) * R_0)) / (2 * (2 - R_0))

    # Index of refraction
    n = (1 + np.sqrt(R_fresnel_0)) / (1 - np.sqrt(R_fresnel_0))

    # Extinction coefficient and absorption coefficient
    kappa = -(wavelength / (4 * np.pi * d_g)) * np.log((R_0 - R_fresnel_0) / (R_fresnel_0 * T_0))
    alpha = 4 * np.pi * kappa / wavelength

    # Refraction angle via Snell's law
    phi_prime = np.arcsin(np.sin(phi) / n)

    # Fresnel reflectivity at phi (unpolarized)
    R_fresnel_phi = 0.5 * (
        ((n * np.cos(phi) - np.cos(phi_prime)) / (n * np.cos(phi) + np.cos(phi_prime))) ** 2
        + ((n * np.cos(phi_prime) - np.cos(phi)) / (n * np.cos(phi_prime) + np.cos(phi))) ** 2
    )

    #  Interface transmissivity at phi
    T_fresnel_phi = 1 - R_fresnel_phi

    # Transmittance and reflectance at phi
    expterm = np.exp(-alpha * d_g / np.cos(phi_prime))
    T_phi = (T_fresnel_phi ** 2 * expterm) / (1 - R_fresnel_phi ** 2 * expterm ** 2)
    R_phi = R_fresnel_phi * (1 + T_phi * expterm)

    return T_phi, R_phi


def calc_layer_TR_coated(T_0, R_0, phi):
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
    coef_T_clear = np.array([-0.0015, 3.355, -3.840, 1.460, 0.0288])
    coef_R_clear = np.array([0.999, -0.563, 2.043, -2.532, 1.054])

    coef_T_bronze = np.array([-0.002, 2.813, -2.341, -0.05725, 0.599])
    coef_R_bronze = np.array([0.997, -1.868, 6.513, -7.862, 3.225])

    # Evaluate polynomials at phi: normalized angular curves of the reference glass,
    # T_phi = T_0 * T_clear_phi and R_phi = R_0 + (1 - R_0) * R_clear_phi
    cos_phi = np.cos(phi)
    cos_pow = np.array([1, cos_phi, cos_phi ** 2, cos_phi ** 3, cos_phi ** 4])  # [cos^0, cos^1, cos^2, cos^3, cos^4]

    T_clear_phi = np.dot(coef_T_clear, cos_pow)
    R_clear_phi = np.dot(coef_R_clear, cos_pow) - T_clear_phi

    T_bronze_phi = np.dot(coef_T_bronze, cos_pow)
    R_bronze_phi = np.dot(coef_R_bronze, cos_pow) - T_bronze_phi

    # Apply to coated glass based on T(0)
    if T_0 > T_0_CLEAR_MIN:
        T_phi = T_0 * T_clear_phi
        R_phi = R_0 * (1 - R_clear_phi) + R_clear_phi
    else:
        T_phi = T_0 * T_bronze_phi
        R_phi = R_0 * (1 - R_bronze_phi) + R_bronze_phi

    return T_phi, R_phi


def _combine(T1, Rf1, Rb1, T2, Rf2, Rb2):
    # T, Rf, Rb of two subsystems in series, subsystem 1 on the exterior side
    denom = 1 - Rf2 * Rb1
    T = T1 * T2 / denom
    Rf = Rf1 + T1 ** 2 * Rf2 / denom
    Rb = Rb2 + T2 ** 2 * Rb1 / denom
    return T, Rf, Rb


def calc_multilayer_TRA(T, Rf, Rb): # recursive calculation of transmittance, reflectance and absorptance of the glazing system

    N = len(T)
    # optical properties of subsystems
    # *_ext2int[k]: layers 0..k, accumulated from the exterior towards the interior
    # *_int2ext[k]: layers k..N-1, accumulated from the interior towards the exterior
    T_ext2int = np.zeros(N)
    T_int2ext = np.zeros(N)
    Rf_ext2int = np.zeros(N)
    Rf_int2ext = np.zeros(N)
    Rb_ext2int = np.zeros(N)
    Rb_int2ext = np.zeros(N)
    A_sys_lyrs = np.zeros(N)

    # exterior -> interior
    T_ext2int[0] = T[0]
    Rf_ext2int[0] = Rf[0]
    Rb_ext2int[0] = Rb[0]

    for k in range(1, N):
        T_ext2int[k], Rf_ext2int[k], Rb_ext2int[k] = _combine(
            T_ext2int[k - 1], Rf_ext2int[k - 1], Rb_ext2int[k - 1], T[k], Rf[k], Rb[k]
        )

    # interior -> exterior
    T_int2ext[N - 1] = T[-1]
    Rf_int2ext[N - 1] = Rf[-1]
    Rb_int2ext[N - 1] = Rb[-1]

    for k in range(N - 2, -1, -1):
        T_int2ext[k], Rf_int2ext[k], Rb_int2ext[k] = _combine(
            T[k], Rf[k], Rb[k], T_int2ext[k + 1], Rf_int2ext[k + 1], Rb_int2ext[k + 1]
        )

    # A
    # 1st layer
    A_sys_lyrs[0] = (1 - T[0] - Rf[0]) + T[0] * Rf_int2ext[1] * (1 - T[0] - Rb[0]) / (1 - Rf_int2ext[1] * Rb_ext2int[0])

    for k in range(1, N - 1):
        A_sys_lyrs[k] = (
            T_ext2int[k - 1] * (1 - T[k] - Rf[k]) / (1 - Rf_int2ext[k] * Rb_ext2int[k - 1])
            + T_ext2int[k] * Rf_int2ext[k + 1] * (1 - T[k] - Rb[k]) / (1 - Rf_int2ext[k + 1] * Rb_ext2int[k])
        )
    # original MATLAB (1-based, old names: Tw_f = T_ext2int, Rfw_b = Rf_int2ext, Rbw_f = Rb_ext2int)
    # for i=2:N-1
    #     Aw_f(i)=Tw_f(i-1)*(1-T(i)-Rf(i))/(1-Rfw_b(i)*Rbw_f(i-1))...
    #            + Tw_f(i)*Rfw_b(i+1)*(1-T(i)-Rb(i))/(1-Rfw_b(i+1)*Rbw_f(i));

    # Nst layer
    A_sys_lyrs[N - 1] = T_ext2int[N - 2] * (1 - T[N - 1] - Rf[N - 1]) / (1 - Rf_int2ext[N - 1] * Rb_ext2int[N - 2])

    T_sys = T_ext2int[-1]
    Rf_sys = Rf_ext2int[-1]
    Rb_sys = Rb_ext2int[-1]

    return T_sys, Rf_sys, Rb_sys, A_sys_lyrs


def _trace_rays(T, Rf, Rb, from_ext, tol, max_iter):
    # Split rays at every layer into absorbed, reflected and transmitted parts
    # until the energy still travelling inside the glazing is below tol.
    #
    # Regions 0..N: region 0 is the exterior, region N the interior and region k
    # the gap between layers k-1 and k. A ray travelling towards the interior in
    # region k hits the front of layer k; a ray travelling towards the exterior in
    # region k hits the back of layer k-1. Rays in the same region and direction
    # are merged at each step, which is exact by linearity and keeps the cost from
    # growing with the number of ray paths.
    N = len(T)
    A_lyrs = np.zeros(N)
    w_ext2int = np.zeros(N + 1)  # weight of rays travelling towards the interior, per region
    w_int2ext = np.zeros(N + 1)  # weight of rays travelling towards the exterior, per region
    if from_ext:
        w_ext2int[0] = 1.0
    else:
        w_int2ext[N] = 1.0
    exit_ext = 0.0  # weight leaving the glazing to the exterior
    exit_int = 0.0  # weight leaving the glazing to the interior

    for _ in range(max_iter):
        if w_ext2int.sum() + w_int2ext.sum() < tol:
            return exit_int, exit_ext, A_lyrs
        hit_front = w_ext2int[:N]  # hits the front of layer k
        hit_back = w_int2ext[1:]   # hits the back of layer k
        A_lyrs += hit_front * (1 - T - Rf) + hit_back * (1 - T - Rb)
        new_int2ext = np.zeros(N + 1)
        new_ext2int = np.zeros(N + 1)
        new_int2ext[:N] = hit_front * Rf + hit_back * T  # reflected from the front, transmitted from the back
        new_ext2int[1:] = hit_front * T + hit_back * Rb  # transmitted from the front, reflected from the back
        exit_ext += new_int2ext[0]
        exit_int += new_ext2int[N]
        new_int2ext[0] = 0.0
        new_ext2int[N] = 0.0
        w_ext2int, w_int2ext = new_ext2int, new_int2ext

    raise RuntimeError(f"Ray tracing did not converge within {max_iter} iterations")


def calc_multilayer_TRA_raytracing(T, Rf, Rb, tol=1e-12, max_iter=100000):
    # Ray-tracing counterpart of calc_multilayer_TRA (Python version of
    # WindowRayTracing_fixed.m), with the same inputs and outputs, for verifying
    # the recursive formulas. tol is the energy left untraced, so it also bounds
    # the error in T_sys + Rf_sys + sum(A_sys_lyrs) = 1.
    T = np.asarray(T, dtype=float)
    Rf = np.asarray(Rf, dtype=float)
    Rb = np.asarray(Rb, dtype=float)

    # light incident from the exterior: transmittance, front reflectance, absorptance
    T_sys, Rf_sys, A_sys_lyrs = _trace_rays(T, Rf, Rb, True, tol, max_iter)
    # light incident from the interior: the part returning to the interior is the back reflectance
    Rb_sys, _, _ = _trace_rays(T, Rf, Rb, False, tol, max_iter)

    return T_sys, Rf_sys, Rb_sys, A_sys_lyrs


def calc_glazing_TRA_dir(T_0, Rf_0, Rb_0, d_g, phi):
    # parameters initialization
    N = len(T_0)  # N glazing layers
    T_phi = np.zeros(N)  # transmittance at an incident angle for each glazing layer
    Rf_phi = np.zeros(N)  # front reflectance at an incident angle for each glazing layer
    Rb_phi = np.zeros(N)  # back reflectance at an incident angle for each glazing layer

    # optical properties for direct radiation
    # calculate optical properties for each glass in a specific incident angle
    for k in range(N):
        if Rf_0[k] != Rb_0[k]:
            T_phi[k], Rf_phi[k] = calc_layer_TR_coated(T_0[k], Rf_0[k], phi)
            _, Rb_phi[k] = calc_layer_TR_coated(T_0[k], Rb_0[k], phi)
        else:
            T_phi[k], Rf_phi[k] = calc_layer_TR_uncoated(T_0[k], Rf_0[k], phi, d_g[k])
            _, Rb_phi[k] = calc_layer_TR_uncoated(T_0[k], Rb_0[k], phi, d_g[k])

    # calculate optical properties for the entire system
    if N == 1:
        T_sys = T_phi.item()
        Rf_sys = Rf_phi.item()
        Rb_sys = Rb_phi.item()
        A_sys_lyrs = 1 - T_sys - Rf_sys
    else:
        T_sys, Rf_sys, Rb_sys, A_sys_lyrs = calc_multilayer_TRA(T_phi, Rf_phi, Rb_phi)

    return T_sys, Rf_sys, Rb_sys, A_sys_lyrs

def calc_glazing_TRA_dif(T_0, Rf_0, Rb_0, d_g):
    # parameters initialization
    N = len(T_0)  # N glazing layers
    T_dif = np.zeros(N)  # hemispherical transmittance for each glazing layer
    Rf_dif = np.zeros(N)  # hemispherical front reflectance for each glazing layer
    Rb_dif = np.zeros(N)  # hemispherical back reflectance for each glazing layer

    # optical properties for diffuse radiation
    phi_deg = np.arange(0, 91, 1)
    T_phi = np.zeros(len(phi_deg))
    Rf_phi = np.zeros(len(phi_deg))
    Rb_phi = np.zeros(len(phi_deg))
    phi = np.deg2rad(phi_deg)

    for k in range(N):
        for m in range(len(phi_deg)):
            if Rf_0[k] != Rb_0[k]:
                T_phi[m], Rf_phi[m] = calc_layer_TR_coated(T_0[k], Rf_0[k], phi[m])
                _, Rb_phi[m] = calc_layer_TR_coated(T_0[k], Rb_0[k], phi[m])
            else:
                T_phi[m], Rf_phi[m] = calc_layer_TR_uncoated(T_0[k], Rf_0[k], phi[m], d_g[k])
                _, Rb_phi[m] = calc_layer_TR_uncoated(T_0[k], Rb_0[k], phi[m], d_g[k])
        weight = 2 * np.cos(phi) * np.sin(phi)
        T_dif[k] = np.trapezoid(T_phi * weight, phi)
        Rf_dif[k] = np.trapezoid(Rf_phi * weight, phi)
        Rb_dif[k] = np.trapezoid(Rb_phi * weight, phi)

    if N == 1:
        T_sys = T_dif.item()
        Rf_sys = Rf_dif.item()
        Rb_sys = Rb_dif.item()
        A_sys_lyrs = 1 - T_sys - Rf_sys
    else:
        T_sys, Rf_sys, Rb_sys, A_sys_lyrs = calc_multilayer_TRA(T_dif, Rf_dif, Rb_dif)

    return T_sys, Rf_sys, Rb_sys, A_sys_lyrs

def calc_glazing_A_dif_int(T_0, Rf_0, Rb_0, d_g):
    # diffuse absorptance of each glazing layer for light incident from the interior,
    # ordered from exterior to interior like the other A_sys_lyrs.
    # light from the interior sees the stack flipped: reverse the layer order and swap
    # the front and back faces of every layer. T and R of the flipped stack equal
    # T_sys, Rb_sys and Rf_sys of the original one, so only A is returned
    _, _, _, A_sys_lyrs = calc_glazing_TRA_dif(np.flip(T_0), np.flip(Rb_0), np.flip(Rf_0), np.flip(d_g))

    return np.flip(A_sys_lyrs)
