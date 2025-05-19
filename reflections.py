import numpy as np
import input as I

#===========================================================================================================
def fresnel_coefficients(i, n, n_inc, n_t):
    i = np.array(i, dtype=float)
    n = np.array(n, dtype=float)

    cos_theta_i = np.dot(i, n)
    sin_theta_i2 = 1 - cos_theta_i**2
    sin_theta_t2 = (n_inc / n_t)**2 * sin_theta_i2

    if sin_theta_t2 > 1.0:
        # Total in_ternal reflection
        return 1,0
    cos_theta_t = np.sqrt(1 - sin_theta_t2)

    rs = (n_inc * cos_theta_i - n_t * cos_theta_t) / (n_inc * cos_theta_i + n_t * cos_theta_t)                      # s-polarized (TE)
    ts = (2 * n_inc * cos_theta_i) / (n_inc * cos_theta_i + n_t * cos_theta_t)
   
    rp = (n_t * cos_theta_i - n_inc * cos_theta_t) / (n_t * cos_theta_i + n_inc * cos_theta_t)                       # p-polarized (TM)
    tp = (2 * n_inc * cos_theta_i) / (n_t * cos_theta_i + n_inc * cos_theta_t)

    R = np.abs(rs)**2
    T = 1 -  R
    return R,T
#===========================================================================================================

#===========================================================================================================
def getAbsorption(rs, tandels, n_diel, ray_lengths):
    abs_total = 0
    nrays_refl = 0
    absr = []
    for i in range(0, len(rs)):
        if i == 26:
            print('stop')
        n_refl = len(rs[i]) - 1
        if n_refl > 0:
            nrays_refl += 1
            ray_len = ray_lengths[i]
            r_te = rs[i]
            absr_i = np.zeros(n_refl + 1)
            An = []
            tandel = tandels[i][0]
            nr = n_diel[i][0]
            kc = I.k0*nr*np.sqrt(1-complex(0,tandel))
            alfa = np.abs(np.imag(kc))
            aux = 0
            for j in range(0, n_refl+1):
                r_tej = r_te[j]
                attenuation = np.exp(-2*alfa*ray_len[j])
                if j == 0:
                    An = np.append(An, r_tej)
                    absr_i[j] = 0
                else:
                    absr_i[j] = An[j-1] - attenuation*An[j-1]
                    An = np.append(An, r_tej*attenuation*An[j-1])

                aux += absr_i[j]
            absr = np.append(absr, aux)
    abs_total = sum(absr)
    p_abs_t = abs_total/I.Nrays
    print(p_abs_t)
    return p_abs_t
#===========================================================================================================
