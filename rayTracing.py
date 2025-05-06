import input as I
import plots
import numpy as np
import pyvista as pv
import gmsh
from scipy.interpolate import LinearNDInterpolator


################################## CLASS DEFINITIONS #######################################################

class Surface:
    def __init__(self, surface, er1, er2, tand1, tand2, isArray, isAperturePlane, isLastSurface, isFirstIx):
        self.surface = surface                          # pyvista PolyData
        self.faces = surface.faces                      # faces of the surface
        self.nodes = surface.points                     # nodes of the surface
        self.er1 = er1                                  # permittivity "below" (closer to the array) the surface
        self.er2 = er2                                  # permittivity "above" the surface
        self.tand1 = tand1                              # loss tangent  inside the surface
        self.tand2 = tand2                              # loss tangent outside the surface
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
        
    # Debugging plot
    if I.plotNormals:
        plots.plot_normals(surface, points, interp_vals, surface_nodes, normals_nodes)

    return interp_vals
#===========================================================================================================



#===========================================================================================================
def snell(i, n, ni, nt):
    # I call it snell but it's actually Heckbert's method
    # i --> unit incident vector
    # n --> unit normal vector to surface
    # ni --> refractive index of the first media
    # nt --> refractive index of the second media

    alpha = ni/nt
    t = np.zeros_like(i)
    for ii in range(len(i)):
        d = np.dot(i[ii],n[ii])
        in_b = 1-((alpha**2)*(1-d**2))
        b = np.sqrt(in_b)
        if in_b<0:
            t[ii] = np.nan
        else:
            t_tmp = alpha*i[ii] + (b-alpha*d)*n[ii]
            t[ii] = t_tmp/np.linalg.norm(t_tmp) 
    return t
#===========================================================================================================

#===========================================================================================================
def distance(A,B):
    dist = np.sqrt(abs(A[:,0]-B[:,0])**2+abs(A[:,1]-B[:,1])**2+abs(A[:,2]-B[:,2])**2)
    return dist
#==========================================================================================================

#===========================================================================================================
def ray(surfaces, direction, sk, Pk, nk, ray_lengths, intersected_faces, next_surf, idx):
    # surfaces --> meshed dome surfaces
    # direction --> DRT / RRT
    # ray_origin --> start point of the ray
    # sk --> vector defining the ray direction
    # Pk --> points where the rays intersect with the surfaces
    # nk --> normals to surfaces in intersection points
    # ray_lengths --> lengths of each section of the ray
    # i --> vector of the incident ray to each surface
    # intersected_faces --> list of indexes of each intersected face by a ray
    # next_surf --> next surface to be intersected

    lastSurf = surfaces[next_surf].isLastSurface
    isArray = surfaces[next_surf].isArray
    isFirstIx = surfaces[next_surf].isFirstIx
    # Find intersections & intersected faces' idx
    current_surf = surfaces[next_surf].surface
    points, ray_idx, faces = current_surf.multi_ray_trace(Pk[idx,:,:], sk[idx,:,:], first_point=True, retry=True)
    idx += 1


    filt_idx = []

    # find only the closest intersection to the source
    ray_origin = np.array([0.0, 0.0, 0.0])
    distances = np.linalg.norm(points - ray_origin, axis=1)
    closest_points = {}
    for i, ray_index in enumerate(ray_idx):
        d = distances[i]
        if ray_index not in closest_points or d < closest_points[ray_index][0]:
            filt_idx.append(i)
            # filtered_points.append(points[i])
            # filtered_indices.append(ray_idx)
            # filtered_faces.append(faces[i])
            closest_points[ray_index] = (d, points[i], faces[i])

    # filtered_points = []
    # filtered_indices = []
    # filtered_faces = []
    filtered_points = np.zeros([len(filt_idx),3])
    filtered_indices = np.zeros(len(filt_idx), dtype=int)
    filtered_faces =  np.zeros(len(filt_idx), dtype=int)
    for i in range(0, len(filt_idx)):
        idxx = filt_idx[i]
        filtered_points[i] = points[idxx]
        filtered_indices[i] = int(ray_idx[idxx])
        filtered_faces[i] = int(faces[idxx])

   
    # for ray_idx, (d, pt, face) in closest_points.items():
    #     filtered_points.append(pt)
    #     filtered_indices.append(int(ray_idx))
    #     filtered_faces.append(int(face))

    Pk = Pk[:,filtered_indices,:]    
    intersected_faces = intersected_faces[filtered_indices,:] # if we want to delete rays that don't intersect
    ray_lengths = ray_lengths[filtered_indices,:]
    nk = nk[:,filtered_indices,:]
    sk = sk[:,filtered_indices,:]

    # 
    Pk[idx,:,:] = filtered_points
    intersected_faces[:,idx-1] = filtered_faces
    # Calculate length
    ray_lengths[:,idx-1] = distance(Pk[idx-1,:,:], filtered_points)
    # Find normals
    normals = find_normals(filtered_points, filtered_faces, current_surf)
    if isFirstIx: normals = -normals
    nk[idx,:,:] = normals #if not isFirstIx else -normals
    # Snell
    if direction == 'DRT':
        n1 = np.sqrt(abs(surfaces[next_surf].er1))  # n_ext
        n2 = np.sqrt(abs(surfaces[next_surf].er2))  # n_int
    else:
        n2 = np.sqrt(abs(surfaces[next_surf].er1))
        n1 = np.sqrt(abs(surfaces[next_surf].er2))

    if direction == 'DRT':
        t = snell(sk[idx-1],normals,n1,n2)
    else:
        t = snell(sk[idx-1],-normals,n1,n2)

    idx_citicalangles = ~np.isnan(np.sum(t, axis=1))
    # Delete rays where critical angles is exceeded
    Pk = Pk[:,idx_citicalangles,:]
    intersected_faces = intersected_faces[idx_citicalangles,:]
    ray_lengths = ray_lengths[idx_citicalangles,:]
    nk = nk[:,idx_citicalangles,:]
    sk[idx,:,:] = t
    sk = sk[:,idx_citicalangles,:]
    # phi_a = phi_a[idx_citicalangles]
    # e_arr = e_arr[idx_citicalangles,:]
    
    # Next surface
    if direction == 'DRT':
        next_surf+=1
    else:
        next_surf-=1
    
    # string = ", ".join([f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})" for point in points])  
    # print('***** Rays intersected at ',  string)  
    if (lastSurf and direction=='DRT') or (isArray and direction=='RRT'):
        return Pk, sk, nk, ray_lengths, intersected_faces
    else:
        return ray(surfaces, direction, sk, Pk, nk, ray_lengths, intersected_faces, next_surf, idx)
#===========================================================================================================


#===========================================================================================================
def DRT (ray_origins, Nx, Ny, sk_0, surfaces):

    N_sections = I.nSurfaces
    N_rays = I.Nrays
    sk = np.zeros([N_sections+1, N_rays, 3])
    sk[0,:,:] = sk_0
    
    direction = 'DRT'

    Pk = np.zeros([N_sections+1, N_rays, 3])
    Pk[0,:,:] = ray_origins
    nk = np.zeros([N_sections+1, N_rays, 3])
    nk[0,:,:] = np.tile(np.array([0,0,1]), (N_rays,1))
    ray_lengths = np.zeros([N_rays, N_sections])
    intersected_faces = np.zeros([N_rays, N_sections])
    next_surf = 1
    idx = 0

    Pk, sk, nk, ray_lengths, idx_intersected_faces = ray(surfaces, direction, sk, Pk, nk, ray_lengths,  intersected_faces, next_surf, idx)
    N_used_rays = np.shape(Pk)[1]

    # Path length calculation
    # path_length = np.zeros((N_used_rays,1), dtype=np.complex128) + (phi_a/I.k0).reshape(-1,1)
    # for ii in range(N_sections):
    #     path_length += ray_lengths[:,ii].reshape(-1,1) * np.sqrt(surfaces[ii+1].er1)

    # Polarization calculation
    # T_tot, e = polarization(Pk, sk, nk, e_arr)

    # if I.plotDRT:
        # plots.plot_rays(surfaces, Pk, N_sections, sk, N_used_rays, direction)

    return Pk, idx_intersected_faces, ray_lengths, nk, sk

#===========================================================================================================



def barycentric_mean(points, faces_intersected, faces_areas, faces_nodes, surface_nodes, normals_nodes):
    """
    Interpolates normals at given points using barycentric coordinates
    inside intersected triangle faces.

    Parameters:
    - points: (N, 3) array of intersected points
    - faces_intersected: (N,) array of triangle indices corresponding to each point
    - faces_areas: (F,) array of face areas
    - faces_nodes: (F*4,) array in PyVista format → [3, i0, i1, i2, 3, i3, i4, i5, ...]
    - surface_nodes: (M, 3) array of coordinates of all mesh nodes
    - normals_nodes: (M, 3) array of normal vectors per node

    Returns:
    - (N, 3) array of interpolated normal vectors at each point
    """
    # Reformat faces_nodes if needed
    if faces_nodes.ndim == 1:
        faces_nodes = faces_nodes.reshape(-1, 4)[:, 1:]

    interpolated_normals = []

    for point, face_idx in zip(points, faces_intersected):
        # Get the 3 node indices of the triangle
        i0, i1, i2 = faces_nodes[face_idx]

        # Get vertex coordinates
        A = surface_nodes[i0]
        B = surface_nodes[i1]
        C = surface_nodes[i2]

        # Get normals at each vertex
        nA = normals_nodes[i0]
        nB = normals_nodes[i1]
        nC = normals_nodes[i2]

        # Compute barycentric coordinates
        v0 = B - A
        v1 = C - A
        v2 = point - A

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if np.abs(denom) < 1e-12:
            # Degenerate triangle — fallback to nA
            interpolated_normals.append(nA)
            continue

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        # Interpolate normal
        normal = u * nA + v * nB + w * nC

        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm

        interpolated_normals.append(normal)

    return np.array(interpolated_normals)
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
    

def fibonacci_sphere(n_rays, randomize=False):
    """
    Devuelve direcciones unitarias uniformemente distribuidas en la esfera
    usando el método de Fibonacci.
    """
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

    # elif typeSrc == 'planew':
    #     origins = 





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


