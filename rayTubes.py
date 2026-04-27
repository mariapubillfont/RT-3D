
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plots as plot
import numpy as np
import pyvista as pv
import readFile as rdFl
import input as I
from numpy import array, real

#===========================================================================================================
def unit(v):
    v = array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Nul vector")
    return v / n 
#===========================================================================================================


#===========================================================================================================
def cos_angle(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#===========================================================================================================


#===========================================================================================================

def field_decomposition(sk, nk, Ex, Ey, Ez):
    # sk, nk: (N, 3)    # Ex, Ey, Ez: (N,)
    
    k_hat = sk / np.linalg.norm(sk, axis=1, keepdims=True)
    n_hat = nk / np.linalg.norm(nk, axis=1, keepdims=True)
    
    # Ei como array (N, 3)
    Ei = np.stack([Ex, Ey, Ez], axis=1)

    # cross por fila
    cross_inner = np.cross(n_hat, k_hat)           # (N, 3)
    cross_outer = np.cross(k_hat, cross_inner)      # (N, 3)
    norm_co = np.linalg.norm(cross_outer, axis=1, keepdims=True)
    e_par = cross_outer / norm_co                   # (N, 3)

    # Evitar división por cero
    valid = norm_co[:, 0] > 1e-10
    e_par = np.zeros_like(cross_outer)
    e_par[valid] = cross_outer[valid] / norm_co[valid]

    Ei_tm = np.einsum('ij,ij->i', Ei, e_par)

    e_perp = -np.cross(e_par, k_hat)
    Ei_te  = np.einsum('ij,ij->i', Ei, e_perp)

    return Ei_te, Ei_tm
#===========================================================================================================



#===========================================================================================================
def get_rayTubes(Pk, sk, theta_t, nk, surfaces, Ak, ff):
    [At_tek, Ar_tek, At_tmk, Ar_tmk] = Ak                                # Unpack coefficients

    # Ray data at source, aperture, and last segment
    Pk_src  = np.array([p[0]  for p in Pk])                              # Source points
    Pk_ap   = np.array([p[1]  for p in Pk])                              # Aperture points
    sk_src  = np.array([s[0]  for s in sk])                              # Source directions
    sk_last = np.array([s[-2] for s in sk])                              # Last directions
    nk_src  = np.array([n[0]  for n in nk])                              # Source normals
    nk_ap   = np.array([n[1]  for n in nk])                              # Aperture normals
    nk_last = np.array([n[-2] for n in nk])                              # Last normal)

    #   Source angles
    theta_all = np.arccos(np.clip(np.real(sk_src[:, 2]), -1.0, 1.0))     # Source theta
    phi_all   = np.arctan2(np.real(sk_src[:, 1]), np.real(sk_src[:, 0])) % (2 * np.pi)  # Source phi

    # Source field in Cartesian coordinates
    Ex_srck, Ey_srck, Ez_srck = rdFl.farfield_to_cartesian(ff, theta_all, phi_all)    # Source field
    A_srck = np.sqrt(np.abs(Ex_srck)**2 + np.abs(Ey_srck)**2 + np.abs(Ez_srck)**2)  # Source amplitude

    # TE/TM decomposition at source
    Ei_tek, Ei_tmk = field_decomposition(sk_src, nk_src, Ex_srck, Ey_srck, Ez_srck)  # Incident TE/TM

    # Cosines between normals and Poyiting vectors of the first and last interfaces
    cos_ak = np.einsum('ij,ij->i', nk_src,  sk_src) / (np.linalg.norm(nk_src,  axis=1) * np.linalg.norm(sk_src,  axis=1))  # cos(alpha_k)
    cos_bk = np.einsum('ij,ij->i', nk_last, sk_last) / (np.linalg.norm(nk_last, axis=1) * np.linalg.norm(sk_last, axis=1))  # cos(beta_k)

    # Triangulation on the two most varying aperture coordinates
    dims = np.argsort(np.ptp(Pk_ap, axis=0))[-2:]                         # Best 2D projection
    triangles = Delaunay(Pk_ap[:, dims]).simplices                        # Triangle indices
    i_idx, j_idx, k_idx = triangles.T                                     # Triangle vertices

    # Triangle vertices at source and aperture
    Ps_src = Pk_src[triangles]                                            # Source triangles
    Ps_ap  = Pk_ap[triangles]                                             # Aperture triangles

    # Triangle areas
    dS_src = 0.5 * np.linalg.norm(np.cross(Ps_src[:, 1] - Ps_src[:, 0], Ps_src[:, 2] - Ps_src[:, 0]), axis=1)  # Source area
    dS_ap  = 0.5 * np.linalg.norm(np.cross(Ps_ap[:, 1]  - Ps_ap[:, 0],  Ps_ap[:, 2]  - Ps_ap[:, 0]),  axis=1)  # Aperture area

    # Aperture centers
    C_ap = Ps_ap.mean(axis=1)                                             # Aperture centroids

    # Mean source direction per triangle
    s_mean = sk_src[triangles].mean(axis=1)                               # Mean direction
    s_mean /= np.linalg.norm(s_mean, axis=1, keepdims=True)               # Normalize direction

    # Triangle angles
    theta_tri = np.arccos(np.clip(np.real(s_mean[:, 2]), -1.0, 1.0))      # Triangle theta
    phi_tri   = np.arctan2(np.real(s_mean[:, 1]), np.real(s_mean[:, 0])) % (2 * np.pi)  # Triangle phi

    # Mean source field per triangle
    Ex_src, Ey_src, Ez_src = rdFl.farfield_to_cartesian(ff, theta_tri, phi_tri)       # Triangle field
    A_src = np.sqrt(np.abs(Ex_src)**2 + np.abs(Ey_src)**2 + np.abs(Ez_src)**2)  # Triangle amplitude

    # Mean cosine factors
    cos_a = cos_ak[triangles].mean(axis=1)                                # Mean cos(alpha)
    cos_b = cos_bk[triangles].mean(axis=1)                                # Mean cos(beta)

    # Aperture amplitude from ray-tube conservation
    A_ap = A_src * np.sqrt(dS_src / (dS_ap * cos_a))                      # Aperture amplitude

    # Mean TE/TM incident field per triangle
    Ei_te = Ei_tek[triangles].mean(axis=1)                                # Mean Ei_te
    Ei_tm = Ei_tmk[triangles].mean(axis=1)                                # Mean Ei_tm

    # Collapse per-ray coefficient histories
    At_tek = np.array([np.sum(a) for a in At_tek], dtype=complex)         # Total At_te per ray
    Ar_tek = np.array([np.sum(a) for a in Ar_tek], dtype=complex)         # Total Ar_te per ray
    At_tmk = np.array([np.sum(a) for a in At_tmk], dtype=complex)         # Total At_tm per ray
    Ar_tmk = np.array([np.sum(a) for a in Ar_tmk], dtype=complex)         # Total Ar_tm per ray

    # Mean coefficients per triangle
    At_te = At_tek[triangles].mean(axis=1)                                # Mean At_te
    Ar_te = Ar_tek[triangles].mean(axis=1)                                # Mean Ar_te
    At_tm = At_tmk[triangles].mean(axis=1)                                # Mean At_tm
    Ar_tm = Ar_tmk[triangles].mean(axis=1)                                # Mean Ar_tm

    if I.plotTubes:
        plot.plot_ray_tubes(Pk_src, sk_src, Pk_ap, triangles, surfaces)   # Plot ray tubes

    return triangles, C_ap, A_ap, A_srck, dS_src, dS_ap, cos_a, cos_b, Ei_te, Ei_tm, [At_te, Ar_te, At_tm, Ar_tm]
#===========================================================================================================


#===========================================================================================================
def get_rayTubes2(Pk, sk, theta_t, nk, surfaces, coefs, ff):
    [At_tek, Ar_tek, At_tmk, Ar_tmk] = coefs

    # --- Extracción vectorizada (sin loops) ---
    Pk_src  = array([Pk_i[0]  for Pk_i in Pk])
    Pk_ap   = array([Pk_i[1]  for Pk_i in Pk])
    sk_src  = array([sk_i[0]  for sk_i in sk])
    sk_ap   = array([sk_i[1]  for sk_i in sk])
    sk_last = array([sk_i[-2] for sk_i in sk])
    nk_src  = array([nk_i[0]  for nk_i in nk])
    nk_ap   = array([nk_i[1]  for nk_i in nk])
    nk_last = array([nk_i[-2] for nk_i in nk])

    Nrays = len(Pk_src)

    # --- Loop 1 vectorizado: ángulos y campo eléctrico ---
    sz_all = real(sk_src[:, 2])
    sy_all = real(sk_src[:, 1])
    sx_all = real(sk_src[:, 0])
    theta_all = np.arccos(np.clip(sz_all, -1, 1))
    phi_all   = np.arctan2(sy_all, sx_all) % (2 * np.pi)

    # get_cartesian_E vectorizado (asumiendo que lo soporta, si no ver nota abajo)
    # Ex_srck, Ey_srck, Ez_srck = ff.get_cartesian_E(theta_all, phi_all)
    Ex_srck, Ey_srck, Ez_srck = rdFl.farfield_to_cartesian(ff, theta_all, phi_all)    # Source field

    # cos_angle vectorizado: dot(a,b) / (|a||b|)
    dot_na_sa = np.einsum('ij,ij->i', nk_src,  sk_src)
    dot_nap_sa = np.einsum('ij,ij->i', nk_ap,  sk_src)
    dot_nl_sl  = np.einsum('ij,ij->i', nk_last, sk_last)
    norm_ns = np.linalg.norm(nk_src,  axis=1)
    norm_sa = np.linalg.norm(sk_src,  axis=1)
    norm_nap = np.linalg.norm(nk_ap,  axis=1)
    norm_nl  = np.linalg.norm(nk_last, axis=1)
    norm_sl  = np.linalg.norm(sk_last, axis=1)
    cos_ak  = dot_na_sa  / (norm_ns  * norm_sa)
    cos_apk = dot_nap_sa / (norm_nap * norm_sa)
    cos_bk  = dot_nl_sl  / (norm_nl  * norm_sl)

    # field_decomposition vectorizado (ver nota abajo si no lo soporta)
    Ei_tek, Ei_tmk = field_decomposition(sk_src, nk_src, Ex_srck, Ey_srck, Ez_srck)

    A_srck = np.sqrt(np.abs(Ex_srck)**2 + np.abs(Ey_srck)**2 + np.abs(Ez_srck)**2)

    # --- Triangulación ---
    # pts_2d = Pk_ap[:, :2]
    # Detectar automáticamente las dos coordenadas con más variación:
    ranges = np.ptp(Pk_ap, axis=0)  # rango de cada coordenada
    dims = np.argsort(ranges)[-2:]  # las 2 con mayor rango
    pts_2d = Pk_ap[:, dims]
    
    tri = Delaunay(pts_2d)
    triangles = tri.simplices
    Ntri = triangles.shape[0]
    i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    # --- Loop 2 vectorizado: operaciones por triángulo ---

    # Puntos fuente y apertura por triángulo: shape (Ntri, 3, 3)
    Ps_src = Pk_src[triangles]   # (Ntri, 3, 3)
    Ps_ap  = Pk_ap[triangles]    # (Ntri, 3, 3)

    # Áreas vectorizadas
    v1_src = Ps_src[:, 1] - Ps_src[:, 0]
    v2_src = Ps_src[:, 2] - Ps_src[:, 0]
    dS_src = 0.5 * np.linalg.norm(np.cross(v1_src, v2_src), axis=1)

    v1_ap = Ps_ap[:, 1] - Ps_ap[:, 0]
    v2_ap = Ps_ap[:, 2] - Ps_ap[:, 0]
    dS_ap  = 0.5 * np.linalg.norm(np.cross(v1_ap, v2_ap), axis=1)

    # Baricentros
    C_ap = Ps_ap.mean(axis=1)   # (Ntri, 3)

    # Normales y direcciones medias por triángulo
    n_mean_s = nk_src[triangles].sum(axis=1)          # (Ntri, 3)
    n_mean_s /= np.linalg.norm(n_mean_s, axis=1, keepdims=True)

    s_mean_s = sk_src[triangles].sum(axis=1)          # (Ntri, 3)
    s_mean_s /= np.linalg.norm(s_mean_s, axis=1, keepdims=True)

    # Ángulos por triángulo
    sz_tri = real(s_mean_s[:, 2])
    sy_tri = real(s_mean_s[:, 1])
    sx_tri = real(s_mean_s[:, 0])
    theta_tri = np.arccos(np.clip(sz_tri, -1, 1))
    phi_tri   = np.arctan2(sy_tri, sx_tri) % (2 * np.pi)

    Ex_src, Ey_src, Ez_src = rdFl.farfield_to_cartesian(ff, theta_tri, phi_tri)


    # Cosenos medios por triángulo
    cos_a  = cos_ak[triangles].mean(axis=1)
    cos_ap = cos_apk[triangles].mean(axis=1)
    cos_b  = cos_bk[triangles].mean(axis=1)

    # Amplitudes
    A_src = np.sqrt(np.abs(Ex_src)**2 + np.abs(Ey_src)**2 + np.abs(Ez_src)**2)
    A_ap  = A_src * np.sqrt(dS_src / (dS_ap * cos_a))

    # field_decomposition vectorizado
    # Ei_te, Ei_tm = field_decomposition(s_mean_s, n_mean_s, Ex_src, Ey_src, Ez_src)
    Ei_te = (Ei_tek[i_idx] + Ei_tek[j_idx] + Ei_tek[k_idx]) / 3
    Ei_tm = (Ei_tmk[i_idx] + Ei_tmk[j_idx] + Ei_tmk[k_idx]) / 3
    lengths = [len(a) for a in At_tek]

   
    # Coeficientes promediados por triángulo
    At_tek_arr = array([np.sum(a) for a in At_tek], dtype=complex)  # (Nrays,)
    Ar_tek_arr = array([np.sum(a) for a in Ar_tek], dtype=complex)
    At_tmk_arr = array([np.sum(a) for a in At_tmk], dtype=complex)
    Ar_tmk_arr = array([np.sum(a) for a in Ar_tmk], dtype=complex)

    # At_tek_arr = At_tek # (Nrays,)
    # Ar_tek_arr = Ar_tek
    # At_tmk_arr = At_tmk
    # Ar_tmk_arr = Ar_tmk

    At_te = (At_tek_arr[i_idx] + At_tek_arr[j_idx] + At_tek_arr[k_idx]) / 3
    Ar_te = (Ar_tek_arr[i_idx] + Ar_tek_arr[j_idx] + Ar_tek_arr[k_idx]) / 3
    At_tm = (At_tmk_arr[i_idx] + At_tmk_arr[j_idx] + At_tmk_arr[k_idx]) / 3
    Ar_tm = (Ar_tmk_arr[i_idx] + Ar_tmk_arr[j_idx] + Ar_tmk_arr[k_idx]) / 3

    if I.plotTubes:
        plot.plot_ray_tubes(Pk_src, sk_src, Pk_ap, triangles, surfaces)

    return triangles, C_ap, A_ap, A_srck, dS_src, dS_ap, cos_a, cos_b, Ei_te, Ei_tm, [At_te, Ar_te, At_tm, Ar_tm]
#===========================================================================================================


#===========================================================================================================
def ray_amplitudes_from_tube_amplitudes(triangles, A_tri, Nrays=None, weights=None):
    """
    Each ray (vertex) is shared by multiple ray tubes (triangles). The ray amplitude is
    defined as the mean of the amplitudes of all triangles incident to that ray. """

    triangles = np.asarray(triangles, dtype=int)
    A_tri = np.asarray(A_tri)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (Ntri, 3)")
    if A_tri.ndim != 1 or A_tri.shape[0] != triangles.shape[0]:
        raise ValueError("A_tri must have shape (Ntri,) and match triangles.shape[0]")

    if Nrays is None:
        Nrays = int(triangles.max()) + 1

    # Accumulators for (weighted) sum and (weighted) count per ray
    A_sum = np.zeros(Nrays, dtype=np.complex128)
    W_sum = np.zeros(Nrays, dtype=np.float64)

    if weights is None:
        # Unweighted mean: each incident triangle contributes equally
        for t, (i, j, k) in enumerate(triangles):
            A = A_tri[t]
            A_sum[i] += A; W_sum[i] += 1.0
            A_sum[j] += A; W_sum[j] += 1.0
            A_sum[k] += A; W_sum[k] += 1.0
    else:
        # Weighted mean: each incident triangle contributes proportionally to weights[t]
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != triangles.shape[0]:
            raise ValueError("weights must have shape (Ntri,) and match triangles.shape[0]")

        for t, (i, j, k) in enumerate(triangles):
            wt = float(w[t])
            if wt <= 0.0 or not np.isfinite(wt):
                continue
            A = A_tri[t] * wt
            A_sum[i] += A; W_sum[i] += wt
            A_sum[j] += A; W_sum[j] += wt
            A_sum[k] += A; W_sum[k] += wt

    # Final mean; mark rays with no incident triangles as NaN
    A_ray = np.full(Nrays, np.nan + 1j*np.nan, dtype=np.complex128)
    valid = W_sum > 0.0
    A_ray[valid] = A_sum[valid] / W_sum[valid]

    # For unweighted case, counts is integer; for weighted, counts is still useful as "total weight"
    counts = W_sum.astype(int) if weights is None else W_sum
    return A_ray, counts
#===========================================================================================================

#===========================================================================================================
def sk_to_angles(s_vec):
    sx, sy, sz = s_vec
    # asumimos |s| = 1
    theta = np.arccos(sz)           # [0, pi]
    phi   = np.arctan2(sy, sx)      # [-pi, pi]
    return theta, phi
#===========================================================================================================

#==================================================
def distance(pointA, pointB):
    return (
        ((pointA[0] - pointB[0]) ** 2) +
        ((pointA[1] - pointB[1]) ** 2) +
        ((pointA[2] - pointB[2]) ** 2)
    ) ** 0.5 # fast sqrt
#==================================================

#=============================================================================
def snell(theta_inc, n1, n2):
    arg = abs(n1)/abs(n2) * np.sin(theta_inc)
    if abs(arg) <= 1:
        theta_ref = np.arcsin(abs(n1) / abs(n2) * np.sin(theta_inc))
    else:
        theta_ref = 0.
    return theta_ref
#=============================================================================


#=============================================================================
def getAngleBtwVectors(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
#=============================================================================


#=============================================================================
def calculateRayTubeAmpl(Pk, Pk1, Pk_ap, Pk_ap1, theta, Gk, Asrc):    #get the amplitude of the E field at the aperture plane
    #Pk - intersection of first ray and array
    #Pk1 - intersection of second ray and array
    #Pk_ap - intersection of first ray and aperture
    #Pk_ap1 - intersection of second ray and aperture
    dLk = distance(Pk, Pk1)#/2                               #ray tube width
    dck_ap = distance(Pk_ap, Pk_ap1)#/2                      #infinitesimal arc length of aperture
    dLk_ap = (dck_ap*np.cos(theta))
    return Asrc*np.sqrt(dLk/dLk_ap)*Gk, dLk, dLk_ap
# =============================================================================

#=============================================================================
def getAmplitude2D( sk_all, nk_all, Pk):
    row = []
    N_rays = len(sk_all)
    Ak_ap = np.zeros(N_rays-2)                                   #amplitude on aperture
    theta_k = np.zeros(N_rays)
    dck = np.zeros(N_rays-2) 
    dLk_src = np.zeros(N_rays-2) 
    dLk_ap = np.zeros(N_rays-2)                                     #infinitesimal arc length of aperture   
    Gk = np.zeros(N_rays) 
    nElement = np.zeros(N_rays)

    Ex_src = np.zeros(N_rays, dtype=complex)
    Ey_src = np.zeros(N_rays, dtype=complex)
    Ez_src = np.zeros(N_rays, dtype=complex)
    sk_src = array([sk_i[0] for sk_i in sk_all])
    Ak_src   = np.zeros(N_rays, dtype=complex)

    for i in range(0, N_rays):                                                   #for each ray
        #nk = [rays[i].normals[nSurfaces*2-2], rays[i].normals[nSurfaces*2-1]]       #normal to surface
        
        nk = nk_all[i][0]
        sk = sk_all[i][0]     
        sx, sy, sz = sk                                                       #poynting vector
        theta_k[i] = getAngleBtwVectors(nk, sk)  
        theta_i = np.arccos(sz)
        phi_i   = np.arctan2(sy, sx)
        if phi_i < 0:
            phi_i += np.pi*2
        Ex_src[i], Ey_src[i], Ez_src[i] = rdFl.get_cartesian_E(theta_i, phi_i)
        if np.isnan(Ex_src[i]):
            print('isnan E field in ray tubes line 42')
        
        Ak_src[i] = np.sqrt(np.abs(Ex_src[i])**2 + np.abs(Ey_src[i])**2 + np.abs(Ez_src[i])**2)
        # Ak_src[i] = 1
        
        if i > 1:                                                                   #exclude first ray, code will handle ray i-1 for each loop
            Pstart1 = Pk[i-2][0]                                     #intersection to the left of ray on array
            Pstart2 = Pk[i][0]                                           #intersection to the right of ray on array   
            dl_src = distance(Pstart1, Pstart2)
            Pap1 = Pk[i-2][1]                             #intersection to the left of ray on aperture
            Pap2 = Pk[i][1]                               #intersection to the left of ray of aperture
            dl_ap = distance(Pap1, Pap2)
            Gk = 1
            Ak_ap[i-2], dLk_src[i-2], dLk_ap[i-2]  = calculateRayTubeAmpl(Pstart1, Pstart2, Pap1, Pap2, theta_k[i-1], Gk, Ak_src[i])
    return Ak_ap, Ak_src, dLk_src, dLk_ap, Ex_src, Ey_src, Ez_src
#=============================================================================


def getA_source(sk0):
    Nrays = len(sk0)
    Ex_src = np.zeros(Nrays, dtype=complex)
    Ey_src = np.zeros(Nrays, dtype=complex)
    Ez_src = np.zeros(Nrays, dtype=complex)
    Et = np.zeros(Nrays)
    for i in range(len(sk0)):                                                      #calculation of the Ex, Ey, Ex of the source. Read file from CST.
            sx, sy, sz = sk0[i]
            # ángulos del rayo i
            theta_i = np.arccos(sz)
            phi_i   = np.arctan2(sy, sx)
            if phi_i < 0:
                phi_i += np.pi*2
            Ex_src[i], Ey_src[i], Ez_src[i] = rdFl.get_cartesian_E(theta_i, phi_i)
            Et[i] = rdFl.get_cartesian_E2(theta_i, phi_i)
            if np.isnan(Ex_src[i]):
                print('isnan')
            # Ex_src[i], Ey_src[i], Ez_src[i] = [1, 1, 1]

    A_src = np.sqrt(np.abs(Ex_src)**2 + np.abs(Ey_src)**2 + np.abs(Ez_src)**2)  
    Pt_src = sum(A_src**2)   
    Pt_src2 = sum(Et**2) 
    return A_src, Pt_src 






