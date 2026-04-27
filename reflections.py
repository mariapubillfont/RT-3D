import numpy as np
import input as I
from numba import njit, prange
from numpy import abs, sum


EPS = 1e-12                                                       # Small threshold

#===========================================================================================================
@njit(cache=True)
def fresnel(i, n, n_inc, n_t):
    cos_i = i[0]*n[0] + i[1]*n[1] + i[2]*n[2]                         # cos(theta_i)
    sin_i2 = 1.0 - cos_i * cos_i                                      # sin^2(theta_i)

    ratio = n_inc / n_t                                               # Refractive index ratio
    k = 1.0 - (ratio * ratio) * sin_i2                                # Argument for sqrt
    cos_t = np.sqrt(k + 0.0j) 
                                          
    den_s = n_inc * cos_i + n_t * cos_t                               # TE denominator
    den_p = n_t * cos_i + n_inc * cos_t                               # TM denominator

    if np.abs(den_s) < EPS:
        rs = 0.0 + 0.0j                                               # Avoid instability
        ts = 0.0 + 0.0j
    else:
        rs = (n_inc * cos_i - n_t * cos_t) / den_s                    # TE reflection
        ts = (2.0 * n_inc * cos_i) / den_s                            # TE transmission

    if np.abs(den_p) < EPS:
        rp = 0.0 + 0.0j                                               # Avoid instability
        tp = 0.0 + 0.0j
    else:
        rp = (n_inc * cos_t - n_t * cos_i) / den_p                    # TM reflection
        tp = (2.0 * n_inc * cos_i) / den_p                            # TM transmission

    return rs, ts, rp, tp, cos_t                                      # Return coefficients
#===========================================================================================================



#===========================================================================================================
def get_Pabs(A_rtubes, Ak, Ak_src, Ak_src_t, Ei_te, Ei_tm, cos_ap, cos_bp, dLk_ap, nrays_refl):
    eta0 = 377.0                                                        # Free-space impedance
    At_te, Ar_te, At_tm, Ar_tm = A_rtubes                               # TE/TM coefficients

    cos_a = abs(cos_ap)                                              # Incident cosine
    cos_b = abs(cos_bp)                                              # Transmitted cosine

    if I.typeSrc == "pw":                                               # Plane wave source, don't need to decompose
        Ei_te_w = Ak                                                   
        Ei_tm_w = Ak                                                   
    else:
        Ei_te_w = Ei_te * Ak                                            # Weighted TE field
        Ei_tm_w = Ei_tm * Ak                                            # Weighted TM field

    Ei_te2 = abs(Ei_te_w)**2                                         # TE incident power term
    Ei_tm2 = abs(Ei_tm_w)**2                                         # TM incident power term

    Si = Ei_te2 / (2 * eta0 * cos_a) + Ei_tm2 * cos_a / (2 * eta0)      # Incident density
    Sr = abs(Ar_te)**2 * Ei_te2 / (2 * eta0 * cos_a) + abs(Ar_tm)**2 * Ei_tm2 * cos_a / (2 * eta0)  # Reflected density
    St = abs(At_te)**2 * Ei_te2 / (2 * eta0 * cos_b) + abs(At_tm)**2 * Ei_tm2 * cos_b / (2 * eta0)  # Transmitted density

    Pi = sum(Si * dLk_ap)                                            # Incident power
    Pr = sum(Sr * dLk_ap)                                            # Reflected power
    Pt = sum(St * dLk_ap)                                            # Transmitted power

    R = Pr / Pi                                                         # Reflection coefficient
    T = Pt / Pi                                                         # Transmission coefficient
    A = 1.0 - R - T                                                     # Absorption coefficient
    A_norm = A*sum(abs(Ak_src)**2) / sum(abs(Ak_src_t)**2)  # Normalized absorption
    # A_norm = A * nrays_refl / I.Nrays                                   # Ray-count normalized absorption

    print(A * sum(abs(Ak_src)**2) / sum(abs(Ak_src_t)**2))  # Power-normalized absorption
    print(A * sum(abs(Ak_src)) / sum(abs(Ak_src_t)))        # Amplitude-normalized absorption
    print(A_norm)                                                       # Normalized absorption

    return A, R, T, A_norm                                              # Return coefficients
#===========================================================================================================


#===========================================================================================================
@njit(cache=True)
def cos_angle(a, b):
    dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]                  # Dot product
    na  = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])         # ||a||
    nb  = np.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])         # ||b||
    if na == 0.0 or nb == 0.0:                               # Avoid division by zero
        return 0.0
    return dot / (na * nb)                                   # cos(angle)
#===========================================================================================================


#==========================================================================================================
def safe_divide(num, den, fill_value=np.nan):
    if abs(den) < EPS or np.isnan(den):
        return fill_value
    return num / den
#==========================================================================================================



#===========================================================================================================
def get_Pabs2D(At_te, Ar_te, At_tm, Ar_tm, Ak, Ei_te, Ei_tm, nk, sk, dLk_ap):
    eta0 = 377.0                                                        # Free-space impedance
    nrays_refl = len(At_te)                                             # Number of reflected rays
    Pi, Pr, Pt = 0.0, 0.0, 0.0                                          # Total powers

    for k in range(1, len(At_te) - 2):                                  # Skip edge rays
        i = k - 1                                                       # Area index
        At_te_k = np.sum(At_te[k])                                      # Total TE transmission
        At_tm_k = np.sum(At_tm[k])                                      # Total TM transmission
        Ar_te_k = np.sum(Ar_te[k])                                      # Total TE reflection
        Ar_tm_k = np.sum(Ar_tm[k])                                      # Total TM reflection

        cos_a = abs(cos_angle(nk[k][0],  sk[k][0]))                     # Incident cosine
        cos_b = abs(cos_angle(nk[k][-2], sk[k][-2]))                    # Transmitted cosine

        Ei_te_k = Ei_te[k] * abs(Ak[i])                                 # Weighted TE field
        Ei_tm_k = Ei_tm[k] * abs(Ak[i])                                 # Weighted TM field

        Ei_te2 = abs(Ei_te_k)**2                                        # TE power term
        Ei_tm2 = abs(Ei_tm_k)**2                                        # TM power term

        Si = Ei_te2 / (2 * eta0 * cos_a) + Ei_tm2 * cos_a / (2 * eta0)  # Incident density
        Sr = abs(Ar_te_k)**2 * Ei_te2 / (2 * eta0 * cos_a) + abs(Ar_tm_k)**2 * Ei_tm2 * cos_a / (2 * eta0)  # Reflected density
        St = abs(At_te_k)**2 * Ei_te2 / (2 * eta0 * cos_b) + abs(At_tm_k)**2 * Ei_tm2 * cos_b / (2 * eta0)  # Transmitted density

        Pi += Si * dLk_ap[i]                                            # Incident power
        Pr += Sr * dLk_ap[i]                                            # Reflected power
        Pt += St * dLk_ap[i]                                            # Transmitted power

    R = Pr / Pi                                                         # Reflection coefficient
    T = Pt / Pi                                                         # Transmission coefficient
    A = 1.0 - R - T                                                     # Absorption coefficient
    A_norm = A * nrays_refl / I.Nrays                                   # Ray-count normalized absorption

    print(A_norm)                                                       
    return A, R, T, A_norm                                              # Return coefficients
#===========================================================================================================
