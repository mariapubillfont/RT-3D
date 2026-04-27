import input as I
import plots
import numpy as np
import reflections as refl
import pyvista as pv
import gmsh
import mesh
from numba import njit, prange
from numpy import array, complex128, append
from scipy.interpolate import LinearNDInterpolator


#===========================================================================================================
def ray(surfaces, Pki, ski, nki, ray_lengthi, cos_ti, er_i, tand_i, r_tei, t_tei, r_tmi, t_tmi, At_tei, Ar_tei, At_tmi, Ar_tmi):
    ni = 0                                                   # number of interfaces crossed by the ray
    next_surf = 1                                            # current surface index to test
    num_refl = 0                                             # number of internal reflected contributions
    num_trans = 0                                            # number of internal transmitted contributions
    ray_len_t = 0.0                                          # accumulated path length used in the phase term
    surf = surfaces[next_surf]                               # current surface object
    kc = I.k0 * surf.er_in * np.sqrt(1 - 1j * surf.tand_in)  # propagation constant in the medium

    while True:
        surf = surfaces[next_surf]                           
        current_surf = surf.surface                          # PyVista surface used for ray tracing
        lastSurf = surf.isLastSurface                        # flag indicating whether this is the last surface
        isFirstIx = (ni == 0)                                # first intersection of this ray
        origin = Pki[ni]                                     # current ray origin
        direction = ski[ni]                                  # current ray direction
        
        chosen_point, chosen_cell = find_closest_intersection(origin, direction, current_surf)          # intersect the ray with the current surface
        if chosen_point is None:
            next_surf += 1
            continue

        er_in = surf.er_in                                   # relative permittivity of incident side
        er_out = surfaces[next_surf-1].er_in                                   # relative permittivity of transmitted side
        tand_in = surf.tand_in                               # loss tangent of incident side
        tand_out = surfaces[next_surf-1].tand_in                             # loss tangent of transmitted side
        n_in = np.sqrt(er_in * (1 - 1j * tand_in))           # refractive index on incident side
        n_out = np.sqrt(er_out * (1 - 1j * tand_out))        # refractive index on transmitted side

        normal = find_normal_cell(chosen_cell, surf)         # outward normal of the intersected cell
        if isFirstIx:
            normal = -normal                                 # flip the normal at the first interface

        Pki[ni + 1] = chosen_point                           # store the new intersection point
        ray_lengthi[ni] = distance(origin, chosen_point)     # store traveled segment length
        nki[ni + 1] = normal                                 # store surface normal at the hit point

        i = ski[ni]                                          # incident direction vector
        n = nki[ni + 1]                                      # interface normal
        r = reflect(i, n)                                    # reflected direction

        if isFirstIx:
            t = snell(i, n, n_out, n_in)                     # refracted direction at the first interface
            ski[ni + 1] = t                                  # propagate transmitted ray after first hit
            r_te, t_te, r_tm, t_tm, cos_t = refl.fresnel(i, n, n_out, n_in)                                                
            kc = I.k0 * n_in                                 # update propagation constant in the dielectric

        else:
            t = snell(i, n, n_in, n_out)                     # refracted direction for subsequent interfaces
            ski[ni + 1] = r                                  # continue following the reflected branch internally
            r_te, t_te, r_tm, t_tm, cos_t = refl.fresnel(i, n, n_in, n_out)                                                
            tand_i[ni] = tand_in                             # store loss tangent of the current medium
            er_i[ni] = np.sqrt(er_in)                        # store refractive index magnitude of the current medium
            kc = I.k0*er_i[ni]*np.sqrt(1-complex(0,tand_i[ni]))

        r_tei[ni] = r_te                                     # store TE reflection coefficient
        t_tei[ni] = t_te                                     # store TE transmission coefficient
        r_tmi[ni] = r_tm                                     # store TM reflection coefficient
        t_tmi[ni] = t_tm                                     # store TM transmission coefficient
        cos_ti[ni] = cos_t                                   # store cosine of transmitted angle

        if r_te == 1:                                        # total internal reflection case
            ni += 1
            continue

        if ni == 0:
            Ar_tei = append(Ar_tei, r_te)                    # first reflected TE contribution
            Ar_tmi = append(Ar_tmi, r_tm)                    # first reflected TM contribution

        elif not lastSurf:
            theta_1 = cos_ti[0]                              # transmission angle at the first interface
            ray_len_t += ray_lengthi[ni]                     # update accumulated internal path length
            phase = np.exp(-1j * kc * ray_len_t * theta_1 * np.abs(theta_1))  # common phase term

            if n[2] >= 0:
                num_trans += 1                               # count transmitted contributions through the slab
                # TE and TM transmitted amplitude contribution
                At_tei = append(At_tei, (-r_tei[0] * r_tei[ni]) ** (num_trans - 1) * (t_tei[0] * t_tei[ni]) * phase)                                           
                At_tmi = append(At_tmi, (-r_tmi[0] * r_tmi[ni]) ** (num_trans - 1) * (t_tmi[0] * t_tmi[ni]) * phase)        

            else:
                num_refl += 1                                # count reflected contributions back toward the source
                # TE and TM reflected amplitude contribution
                Ar_tei = append(Ar_tei, t_tei[0] * t_tei[ni] * r_tei[ni] * (-r_tei[0] * r_tei[ni]) ** (num_refl - 1) * phase)                                          
                Ar_tmi = append(Ar_tmi, t_tmi[0] * t_tmi[ni] * r_tmi[ni] * (-r_tmi[0] * r_tmi[ni]) ** (num_refl - 1) * phase)


        ni += 1                                              # +1 in the interface counter
        surfaces[next_surf].isFirstIx = False                # mark that the first interaction has already happened

        if ni >= I.Nrefl:
            next_surf += 1                                   # move to the next surface after the max number of reflections
            ski[ni] = t                                      # continue with the transmitted direction

        if lastSurf:
            return Pki, ski, nki, ray_lengthi, cos_ti, er_i, tand_i, r_tei, t_tei, r_tmi, t_tmi, At_tei, Ar_tei, At_tmi, Ar_tmi           
#===========================================================================================================


#===========================================================================================================
def DRT (ray_origins, sk_0, surfaces):
    N_rays = I.Nrays
    N_int_max = int(I.nStruct * I.Nrefl + 1)
    n_int = np.zeros(N_rays, dtype=int)                             # real number of intersections for each ray
 
    Pk = np.zeros((N_rays, N_int_max + 1, 3), dtype=np.float64)         # ray points
    sk = np.zeros((N_rays, N_int_max + 1, 3), dtype=np.complex128)      # ray directions (Poyting vector) per intersection, for each ray
    nk = np.zeros((N_rays, N_int_max + 1, 3), dtype=np.complex128)      # surface normals

    ray_lengths = np.zeros((N_rays, N_int_max), dtype=np.float64)   # ray lengths in [m]       
    cos_t = np.zeros((N_rays, N_int_max), dtype=np.complex128)      # cosine of transmission angles in [rad]
    tand = np.zeros((N_rays, N_int_max), dtype=np.complex128)       # loss tangent (tanδ)
    er_nxt = np.zeros((N_rays, N_int_max), dtype=np.complex128)     # refractive index

    r_te = np.zeros((N_rays, N_int_max), dtype=np.complex128)       # TE reflection coeff
    t_te = np.zeros((N_rays, N_int_max), dtype=np.complex128)       # TE transmission coeff
    r_tm = np.zeros((N_rays, N_int_max), dtype=np.complex128)       # TM reflection coeff
    t_tm = np.zeros((N_rays, N_int_max), dtype=np.complex128)       # TM transmission coeff

    At_te = [[] for _ in range(N_rays)]                             # TE transmitted amplitude
    Ar_te = [[] for _ in range(N_rays)]                             # TE reflected amplitude
    At_tm = [[] for _ in range(N_rays)]                             # TM transmitted amplitude
    Ar_tm = [[] for _ in range(N_rays)]                             # TM reflected amplitude

    for i in range(0, N_rays):
        Pk[i, 0] = ray_origins[i]
        sk[i, 0] = sk_0[i]
        nk[i, 0] = np.array([0.0, 0.0, 1.0], dtype=np.complex128)
        if i == 10:
            print('stop at 10th ray')
        (
            Pk[i], sk[i], nk[i], ray_lengths[i], cos_t[i], er_nxt[i], tand[i],
            r_te[i], t_te[i], r_tm[i], t_tm[i],
            At_te[i], Ar_te[i], At_tm[i], Ar_tm[i]
        ) = ray(
            surfaces, Pk[i], sk[i], nk[i], ray_lengths[i], cos_t[i], er_nxt[i], tand[i], 
            r_te[i], t_te[i], r_tm[i], t_tm[i],
            At_te[i], Ar_te[i], At_tm[i], Ar_tm[i]
        )

    # keep only rays intersected with the dielectric (at least 2 interactions → len(Pk) > 2)
    # valid_idx = [i for i, P in enumerate(Pk) if len(P) > 2]
    valid_mask = np.linalg.norm(Pk[:, 2:, :], axis=2).any(axis=1)             # Has dielectric interactions
    valid_idx  = np.where(valid_mask)[0] 
    ray_id = valid_idx[:]  

    # filter all data using valid indices
    Pk = Pk[valid_idx]
    sk = sk[valid_idx]
    nk = nk[valid_idx]
    ray_lengths = ray_lengths[valid_idx]
    cos_t = cos_t[valid_idx]
    er_nxt = er_nxt[valid_idx]
    tand = tand[valid_idx]
    r_te        = [r_te[i]        for i in valid_idx]
    t_te        = [t_te[i]        for i in valid_idx]
    r_tm        = [r_tm[i]        for i in valid_idx]
    t_tm        = [t_tm[i]        for i in valid_idx]
    At_te       = [At_te[i]       for i in valid_idx]
    Ar_te       = [Ar_te[i]       for i in valid_idx]
    At_tm       = [At_tm[i]       for i in valid_idx]
    Ar_tm       = [Ar_tm[i]       for i in valid_idx]
  
    return ray_id, Pk, nk, sk, er_nxt, tand, ray_lengths, cos_t, r_te, t_te, r_tm, t_tm, At_te, Ar_te, At_tm, Ar_tm
#===========================================================================================================

#===========================================================================================================
def find_closest_intersection(origin, direction, current_surf, tol=1e-6):
    end_point = origin + 1e3 * np.real(direction)                 # define a far point along the ray direction
    points, id_cells = current_surf.ray_trace(origin, end_point)  # compute all intersections with the surface

    if len(points) == 0:                                     # no intersection found
        return None, None

    distances = np.linalg.norm(points - origin, axis=1)      # compute distance from origin to each intersection
    valid_mask = distances > tol                             # remove self-intersections / numerical noise

    if not np.any(valid_mask):                               # no valid intersections after filtering
        return None, None

    valid_points = points[valid_mask]                        # keep only valid intersection points
    valid_cells = np.asarray(id_cells)[valid_mask]           # keep corresponding cell indices
    valid_distances = distances[valid_mask]                  # keep corresponding distances
    min_idx = np.argmin(valid_distances)                     # find the closest valid intersection

    return valid_points[min_idx], valid_cells[min_idx]       # return closest point and its cell index
#===========================================================================================================


#===========================================================================================================
def find_normal_cell(face_intersected, surf_obj):
    # return normalized normal of a face
    n = surf_obj.cell_normals[face_intersected]
    n /= np.linalg.norm(n)
    return n
#===========================================================================================================


#===========================================================================================================
@njit(cache=True, fastmath=True)
def snell(i, n, ninc, nt):
## I call it snell but it's actually Heckbert's method
## i --> unit incident vector
## n --> unit normal vector to surface
## ni --> refractive index of the first media
## nt --> refractive index of the second media
    alpha = ninc / nt
    d = np.dot(i, n)
    in_b = 1 - (alpha**2) * (1 - d**2)
    b = np.sqrt(in_b)
    t_complex = alpha * i + (b - alpha * d) * n
    return t_complex
#===========================================================================================================


#===========================================================================================================
@njit(cache=True, fastmath=True)
def reflect(i, n):
    r = i - 2 * np.dot(i, n) * n
    return r / np.linalg.norm(r)
#===========================================================================================================


#===========================================================================================================
@njit(cache=True, fastmath=True)
def distance(A, B):
    return np.linalg.norm(A - B)
#==========================================================================================================

