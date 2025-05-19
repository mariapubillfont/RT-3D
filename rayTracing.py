import input as I
import plots
import numpy as np
import reflections as refl
import pyvista as pv
import gmsh

from scipy.interpolate import LinearNDInterpolator


################################## CLASS DEFINITIONS #######################################################

class Surface:
    def __init__(self, surface, er_out, er_in, tand_out, tand_in, isArray, isAperturePlane, isLastSurface, isFirstIx):
        self.surface = surface                          # pyvista PolyData
        self.faces = surface.faces                      # faces of the surface
        self.nodes = surface.points                     # nodes of the surface
        self.er_in = er_in                                  # permittivity "below" (closer to the array) the surface
        self.er_out = er_out                                  # permittivity "above" the surface
        self.tand_in = tand_in                              # loss tangent  inside the surface
        self.tand_out = tand_out                              # loss tangent outside the surface
        self.isArray = isArray                          # true if the surface is the array
        self.isAperturePlane = isAperturePlane          # true if the surface is the aperture plane
        self.isLastSurface = isLastSurface              # true if the surface is the last surface of the radome
        self.isFirstIx = isFirstIx

############################################################################################################

#===========================================================================================================
def find_normals(points, faces_intersected, surface):
    surface_nodes = surface.points
    normals_nodes = surface.compute_normals(cell_normals=False)['Normals']
    # if np.all(normals_nodes[:,2]<0):
    #     surface.flip_normals()
    #     normals_nodes = surface.compute_normals(cell_normals=False)['Normals']
    # surface_nodes = surface_nodes[normals_nodes[:,2]>0]
    # normals_nodes = normals_nodes[normals_nodes[:,2]>0]

    same_val1 = np.all(abs(surface_nodes) < abs(surface_nodes[0,:])+1e-3, axis = 0)                         # checking if all the points have the same value in the  
    same_val2 = np.all(abs(surface_nodes) > abs(surface_nodes[0,:])-1e-3, axis = 0)                         # z-axis (flat surface) bc the interpolator needs to treat 
    same_val = same_val1[2] and same_val2[2]                                                                # this case separately
    if same_val:
        surface.flip_normals()
        normals_nodes = surface.compute_normals(cell_normals=False)['Normals']
        f = LinearNDInterpolator(surface_nodes[:,0:2], normals_nodes.reshape([len(normals_nodes),3]))
        interp_vals = f(points[:,0:2])
    else:
        faces_areas = surface.compute_cell_sizes(length=False, volume=False).cell_data['Area']
        faces_nodes = surface.faces
        interp_vals = barycentric_mean(points, faces_intersected, faces_areas, faces_nodes, surface_nodes, normals_nodes)
    # CORRECCIÓN: Forzar normales en las tapas del cilindro
    z_min = surface_nodes[:, 2].min()
    z_max = surface_nodes[:, 2].max()
    tol = 1e-5  # tolerancia para identificar si está en la tapa

    for i, pt in enumerate(points):
        if abs(pt[2] - z_min) < tol:
            interp_vals[i] = np.array([0, 0, -1])
        elif abs(pt[2] - z_max) < tol:
            interp_vals[i] = np.array([0, 0, 1])
        
    # Debugging plot
    if I.plotNormals:
        plots.plot_normals(surface, points, interp_vals, surface_nodes, normals_nodes)

    return interp_vals
#===========================================================================================================

#===========================================================================================================
def snell(i, n, ninc, nt):
## I call it snell but it's actually Heckbert's method
## i --> unit incident vector
## n --> unit normal vector to surface
## ni --> refractive index of the first media
## nt --> refractive index of the second media
    i = np.array(i, dtype=float)
    n = np.array(n, dtype=float)
    alpha = ninc / nt
    d = np.dot(i, n)
    in_b = 1 - (alpha**2) * (1 - d**2)
    if in_b < 0:
        return np.full_like(i, np.nan)
    else:
        b = np.sqrt(in_b)
        t = alpha * i + (b - alpha * d) * n
        return t / np.linalg.norm(t)
#===========================================================================================================

def reflect(i, n):
    i = np.array(i, dtype=float)
    n = np.array(n, dtype=float)

    r = i - 2 * np.dot(i, n) * n
    return r / np.linalg.norm(r)


#===========================================================================================================
def distance(A,B):
    dist = np.sqrt(abs(A[:][0]-B[:][0])**2+abs(A[:][1]-B[:][1])**2+abs(A[:][2]-B[:][2])**2)
    return dist
#==========================================================================================================



    # surfaces --> meshed dome surfaces
    # ray_origin --> start point of the ray
    # sk --> vector defining the ray direction
    # Pk --> points where the rays intersect with the surfaces
    # nk --> normals to surfaces in intersection points
    # ray_lengths --> lengths of each section of the ray
    # i --> vector of the incident ray to each surface
    # intersected_faces --> list of indexes of each intersected face by a ray
    # next_surf --> next surface to be intersected
#===========================================================================================================
def ray(surfaces, sk, Pk, nk, ray_lengths, intersected_faces, next_surf, idx, r_tes, tandels, n_diel):
    lastSurf = surfaces[next_surf].isLastSurface
    isArray = surfaces[next_surf].isArray
    isFirstIx = surfaces[next_surf].isFirstIx
    # Find intersections & intersected faces' idx
    current_surf = surfaces[next_surf].surface

    origins = np.array([ray[-1] for ray in Pk])
    directions = np.array([d[-1] for d in sk])
    points, ray_idx, faces = current_surf.multi_ray_trace(origins, directions, first_point=False, retry=True)
    

    # Solo comparar los rayos que realmente intersectaron algo
    valid_origins = origins[ray_idx]
    distances = np.linalg.norm(points - valid_origins, axis=1)

    # # Filtrar los que tienen distancia mayor a un umbral (ej. 1e-3)
    # threshold = 1e-3
    # valid_mask = distances > threshold

    # # Aplicar filtro a todos los arrays
    # points = points[valid_mask]
    # ray_idx = ray_idx[valid_mask]
    # faces = faces[valid_mask]
    
    idx += 1
     # Snell
    n_in = np.sqrt(abs(surfaces[next_surf].er_in))  # n_ext
    n_out = np.sqrt(abs(surfaces[next_surf].er_out))  # n_int
    normals = find_normals(points, faces, current_surf)
    if isFirstIx: normals = -normals

    for pt, i_ray, i_face, i_nk in zip(points, ray_idx, faces, normals):

        if i_ray == 36:
            print('stop')
            aux1, aux2 = current_surf.ray_trace([0,0,0], sk[i_ray][-1])
                    # Pk[i_ray].append([pt[0], pt[1], round(pt[2], 2)])
        Pk[i_ray].append(pt.tolist())
        intersected_faces[i_ray].append(i_face)
        ray_lengths[i_ray].append(distance(Pk[i_ray][idx-1], pt)) 
        nk[i_ray].append(i_nk)
        i = sk[i_ray][-1]
        n = nk[i_ray][-1]
        t = snell(i, n, n_out, n_in)
        r = reflect(i,n)

        if isFirstIx:
            sk[i_ray].append(t)
            ri, ti = refl.fresnel_coefficients(i, n, n_out, n_in)
            r_tes[i_ray].append(ti)
            
        else:
            sk[i_ray].append(r)
            sameDir = np.dot(n,i)
            if sameDir > 0:                                     #n and i have the same direction
                ri, ti = refl.fresnel_coefficients(i, n, n_in, n_out)
            else:
                ri, ti = refl.fresnel_coefficients(i, n, n_in, n_out)
            r_tes[i_ray].append(ri)
        
        tandels[i_ray].append(surfaces[next_surf].tand_in)
        n_diel[i_ray].append(np.sqrt(surfaces[next_surf].er_in))    
        
    
    surfaces[next_surf].isFirstIx = False
    if idx >+ I.maxRefl: 
        next_surf += 1
        idx = 1
    # next_surf += 1
    if lastSurf:
        return Pk, sk, nk, r_tes, tandels, n_diel, ray_lengths
    else:
        return ray(surfaces, sk, Pk, nk, ray_lengths, intersected_faces, next_surf, idx, r_tes, tandels, n_diel)
#===========================================================================================================


#===========================================================================================================
def DRT (ray_origins, Nx, Ny, sk_0, surfaces):
    N_rays = I.Nrays

    Pk = [[ray_origins[i].tolist()] for i in range(N_rays)]                            #List of N_rays elements, 1 element per each ray
    sk = [[] for _ in range(N_rays)] 
    for i in range(N_rays):
        sk[i].append(sk_0[i])
    nk = [[[0.0, 0.0, 1.0]] for _ in range(N_rays)] 
    ray_lengths = [[] for _ in range(N_rays)] 
    intersected_faces = [[] for _ in range(N_rays)] 
    r_tes = [[] for _ in range(N_rays)] 
    tandels = [[] for _ in range(N_rays)] 
    n_diel = [[] for _ in range(N_rays)]
    
    next_surf = 1
    idx = 0

    Pk, sk, nk, r_tes, tandels, n_diel, ray_lengths  = ray(surfaces, sk, Pk, nk, ray_lengths,  intersected_faces, next_surf, idx, r_tes, tandels, n_diel)


    return Pk, nk, sk, r_tes, tandels, n_diel, ray_lengths
#===========================================================================================================


 #===========================================================================================================
def barycentric_mean(points, faces_intersected, faces_areas, faces_nodes, surface_nodes, normals_nodes):
    # Reshape faces_nodes from PyVista format: (F*4,) → (F, 3)
    n_faces = len(faces_areas)
    faces = faces_nodes.reshape((n_faces, 4))[:, 1:]  # Drop the leading '3' in each face

    interpolated_normals = np.zeros_like(points)

    for i, (point, face_idx) in enumerate(zip(points, faces_intersected)):
        i0, i1, i2 = faces[face_idx]                                # Get node indices for the triangle
        A = surface_nodes[i0]                                       # Get vertex positions
        B = surface_nodes[i1]
        C = surface_nodes[i2]
        nA = normals_nodes[i0]                                      # Get normals at each vertex
        nB = normals_nodes[i1]
        nC = normals_nodes[i2]

        v0 = B - A                                                   # Compute vectors for barycentric coordinates
        v1 = C - A
        v2 = point - A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if np.abs(denom) < 1e-12:    
            interpolated_normals[i] = nA                            # Degenerate triangle — fallback to vertex normal
            continue

        v = (d11 * d20 - d01 * d21) / denom                         # Barycentric coordinates
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        normal = u * nA + v * nB + w * nC                           # Interpolated normal
        
        norm = np.linalg.norm(normal)                               # Normalize
        if norm > 1e-12:
            normal /= norm
        interpolated_normals[i] = normal
    return interpolated_normals
 #===========================================================================================================


 #===========================================================================================================
def shootRays(Nx, Ny, Lx, Ly, typeSrc):
    n_theta = int(I.Ntheta)
    n_phi = int(I.Nphi)
    Nrays = int(n_theta*n_phi*Nx*Ny)
    origins = np.zeros([n_theta*n_phi, 3])

    if typeSrc[0] == 'iso':
        sk0 = []
        phi = np.linspace(I.rangePhi[0], I.rangePhi[1], n_phi )
        theta = np.linspace(I.rangeTheta[0], I.rangeTheta[1], n_theta)
        sk0 = np.zeros([Nrays, 3])
        ij = 0
        for i in range(0, n_theta):
            for j in range(0, n_phi):    
                x = np.sin(theta[i]) * np.cos(phi[j])
                y = np.sin(theta[i]) * np.sin(phi[j])
                z = np.cos(theta[i])
                sk0[ij] = np.vstack((x, y, z)).T  # shape (n_rays, 3)
                ij += 1
    return origins, sk0
 #===========================================================================================================
    

 #===========================================================================================================
def fibonacci_sphere(n_rays, randomize=False):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * n_rays

    points = []
    offset = 2.0 / n_rays
    increment = np.pi * (3.0 - np.sqrt(5))  # ángulo dorado

    for i in range(n_rays):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = ((i + rnd) % n_rays) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])  # ya está normalizado
    return np.array(points)
 #===========================================================================================================






# #===========================================================================================================
# def polarization(Pk, sk, nk, e_arr):
#     # e --> unit vector in the e-field direction
#     # v_perp --> unit vector perpendicular to the incidence plane
#     # e_perp --> projection of the e-field on v_perp
#     # v_paral --> unit vector parallel to the incidence plane
#     # e_paral --> projection of the e-field on v_paral
#     e = np.zeros([I.nSurfaces+1,np.shape(e_arr)[0],np.shape(e_arr)[1]])
#     e[0] = e_arr - np.tile(np.diagonal(e_arr@np.transpose(sk[0,:,:])).reshape(-1,1),(1,3))*sk[0,:,:]
#     e[0] = e[0]/np.tile(np.sqrt(e[0,:,0]**2+e[0,:,1]**2+e[0,:,2]**2).reshape(-1,1), (1,3))
#     # dot = np.diagonal(e_arr@np.transpose(sk[0,:,:]))
#     # insk = np.tile(dot.reshape(100,1),(1,3))*sk[0,:,:]

#     T_tot = np.ones(np.shape(e_arr)[0])
#     for i in range(I.nSurfaces):
#         # Compute parallel and perpendicular components of the e-field to the incident plane
#         v_perp = np.cross(nk[i+1,:,:],sk[i+1,:,:])
#         e_perp = np.tile(np.diagonal(e[i]@np.transpose(v_perp)).reshape(-1,1),(1,3))*v_perp
#         e_paral = e[i]-e_perp
#         # Debugging lines
#         # sum = e_perp+e_paral
#         # test = np.sqrt(sum[:,0]**2+sum[:,1]**2+sum[:,2]**2)
#         # Compute Fresnel transmisison coeffs
#         cos_theta_i = np.diagonal(nk[i+1,:,:]@np.transpose(sk[i,:,:]))
#         theta_i = np.arccos(cos_theta_i)
#         sin_theta_t = np.cross(nk[i+1,:,:],sk[i+1,:,:])
#         sin_theta_t = np.sqrt(sin_theta_t[:,0]**2+sin_theta_t[:,1]**2+sin_theta_t[:,2]**2)
#         theta_t = np.arcsin(sin_theta_t)
#         T_perp = (2*cos_theta_i*sin_theta_t)/(np.sin(theta_i+theta_t))
#         T_paral = (2*cos_theta_i*sin_theta_t)/(np.sin(theta_i+theta_t)*np.cos(theta_i-theta_t))
#         # Case where incident angle is 0
#         idxs1 = np.where(theta_i>-1e-3,1,0)
#         idxs2 = np.where(theta_i<1e-3,1,0)
#         idxs = np.where(idxs1*idxs2)
#         T_perp[idxs] = (2*np.sqrt(np.real(I.er[i])))/(np.sqrt(np.real(I.er[i]))+np.sqrt(np.real(I.er[i+1])))
#         T_paral[idxs] = T_perp[idxs]
#         # Apply coeffs
#         e_perp *= np.tile(T_perp.reshape(-1,1), (1,3))
#         e_paral *= np.tile(T_paral.reshape(-1,1), (1,3))    

#         sum = e_perp+e_paral
#         T_tot *= np.sqrt(sum[:,0]**2+sum[:,1]**2+sum[:,2]**2)

#         e[i+1] = sum/np.tile(np.sqrt(sum[:,0]**2+sum[:,1]**2+sum[:,2]**2).reshape(-1,1),(1,3))

#     return T_tot, e
# #===========================================================================================================


# def RRT(surfaces, ap_ray_origins, sk0):
#     # Inputs
#     direction = 'RRT'
#     N_sections = I.nSurfaces + 1
#     N_rays = I.N_rrt
#     sk = np.zeros([N_sections+1, N_rays, 3])
#     sk[0,:,:] = np.tile(sk0, (N_rays,1))
#     Pk = np.zeros([N_sections+1, N_rays, 3])
#     Pk[0,:,:] = ap_ray_origins.T
#     nk = np.zeros([N_sections+1, N_rays, 3])
#     nk[0,:,:] = np.tile(np.array([0,0,1]), (N_rays,1))
#     ray_lengths = np.zeros([N_rays, N_sections])
#     phi_ap = np.zeros([N_rays,1])
#     e_ap = np.zeros([N_rays,3])
#     intersected_faces = np.zeros([N_rays, N_sections])
#     next_surf = I.nSurfaces
#     idx = 0

#     # GO
#     Pk, sk, nk, ray_lengths, phi_ap, e_ap, idx_intersected_faces = ray(surfaces, direction, sk, Pk, nk, ray_lengths, phi_ap, e_ap, intersected_faces, next_surf, idx)
#     N_used_rays = np.shape(Pk)[1]

#     # Interpolate poynting vectors
#     f_inc_vectors = LinearNDInterpolator(Pk[-1][:,0:2], -sk[-2])

#     # Calculate phases
#     phases_rrt = np.zeros([N_used_rays])
#     path_length = np.zeros([N_used_rays])
#     for ii in range(N_sections):
#         path_length += ray_lengths[:,ii] * np.sqrt(abs(surfaces[N_sections-ii].er1))
#     phases_rrt =  I.k0 * path_length    

#     # Interpolate phases
#     f_phases_rrt = LinearNDInterpolator(Pk[-1][:,0:2], phases_rrt)

#     # Plot
#     # if I.plotRRT:
#         # plots.plot_rays(surfaces, Pk, N_sections, sk, N_used_rays, direction)

#     return f_inc_vectors, f_phases_rrt, path_length

# #===========================================================================================================


