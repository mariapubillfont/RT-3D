import time                                                
import numpy as np                                         
import input as I                                          
import mesh                                                
import plots                                               
import reflections as refl                                 
import rayTracing as rt                                    
import rayTubes as tubes                                   
import readFile as rdFl                                    

start = time.perf_counter()                                # start global timer

# =========================================
# Input parameters
# =========================================
k0 = I.k0                                                  # free-space wavenumber
wv = I.wv                                                  # wavelength
D = I.D                                                    # aperture diameter
p = I.p                                                    # x-values
Nx = I.Nx                                                  # number of sources along x
Ny = I.Ny                                                  # number of sources along y
bodies = I.bodies                                          # !!!!!! body definitions
# typeSurface = I.typeSurface                                # surface type
nStruct = I.nStruct                                        # number of structures
Nrays = I.Nrays                                            # total number of launched rays
extra = 1e-6                                               # small offset to avoid boundary issues

surfaces = mesh.create_surfaces()                          # create all simulation surfaces


# =========================================
# Source generation
# =========================================
if I.typeSrc == 'pw':                                               # plane wave in 3D
    x = np.linspace(-I.Lx/2 + extra, I.Lx/2 - extra, I.Nx)          # source sampling points along x
    y = np.linspace(-I.Ly/2 + extra, I.Ly/2 - extra, I.Ny)          # source sampling points along y
    X, Y = np.meshgrid(x, y)                                        # 2D source grid
    Z = np.zeros_like(X)                                            # z = 0 array plane
    origins = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)   # ray origins on the source plane

    theta_src = np.deg2rad(I.theta_pw)                              # plane-wave elevation angle
    phi_src = np.deg2rad(I.phi_pw)                                  # plane-wave azimuth angle
    
    k_dir = np.array([np.sin(theta_src) * np.cos(phi_src), np.sin(theta_src) * np.sin(phi_src), np.cos(theta_src)])
    sk0 = np.tile(k_dir, (Nrays, 1))                                # same direction for all rays

elif I.typeSrc == '2D':                                             # plane wave in 2D
    x = np.linspace(-I.Lx/2 + extra, I.Lx/2 - extra, Nrays)         # ray origins along x
    y = np.zeros(Nrays)                                             # y = 0
    z = np.zeros(Nrays)                                             # z = 0
    origins = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)   # 2D line source

    theta_src = np.linspace(-np.pi, np.pi - 2*np.pi/(Nrays - 1), Nrays)  # angular sweep in theta
    phi_src = np.zeros(Nrays)                                       # phi = 0 for all rays

    x_sk = np.sin(theta_src) * np.cos(phi_src)                      # x component of ray directions
    y_sk = np.sin(theta_src) * np.sin(phi_src)                      # y component of ray directions
    z_sk = np.cos(theta_src)                                        # z component of ray directions
    sk0 = np.column_stack((x_sk, y_sk, z_sk))                       # initial ray directions

else:
    origins = mesh.sphere_sampling(Nrays, I.Lx)                     # ray origins distributed on a sphere of radius equal to the source size
    sk0 = mesh.sphere_sampling(Nrays, 1)                            # initial ray directions uniformly distributed on a unit sphere   

if I.plotSurf:
    plots.plotSurfaces(surfaces, origins, sk0)                      # plot geometry and launched rays

end = time.perf_counter()
print(f"Create and plot surfaces: {end - start:.4f} sec.")          # timing for geometry/source setup



# =========================================
# Source E-field interpolation
# =========================================
start = time.perf_counter()

sx = sk0[:, 0]                                                      # x component of ray directions
sy = sk0[:, 1]                                                      # y component of ray directions
sz = sk0[:, 2]                                                      # z component of ray directions

theta_ff = np.arccos(sz)                                            # spherical theta for each ray direction
phi_ff = np.mod(np.arctan2(sy, sx), 2 * np.pi)                      # spherical phi in [0, 2pi)

ff = rdFl.build_farfield_interpolator("Sources/wvgd_src_lin.txt")               # source far-field interpolator
Ex_src, Ey_src, Ez_src = rdFl.farfield_to_cartesian(ff, theta_ff, phi_ff)       # interpolate source E-field in Cartesian components

A_src_t = np.sqrt(np.abs(Ex_src)**2 + np.abs(Ey_src)**2 + np.abs(Ez_src)**2)    # E-field amplitude

end = time.perf_counter()
print(f"Get E-field: {end - start:.4f} sec.")                        # timing for source field interpolation



# =========================================
# Ray tracing
# =========================================
start = time.perf_counter()

(
    ray_ids, Pk, nk, sk, ndiel, tandels, ray_lengths, theta_ts,
    r_tes, t_tes, r_tms, t_tms, At_te, Ar_te, At_tm, Ar_tm
) = rt.DRT(origins, sk0, surfaces)                                   # direct ray tracing

end = time.perf_counter()
print(f"Calculate ray tracing: {end - start:.4f} sec.")              # timing for ray tracing stage

if I.plotDRT:
    plots.plotDRT(surfaces, Pk)                                  # direct RT plot

if I.plotNormals:
    plots.plot_normals(surfaces, np.real(nk), np.real(Pk))           # Plot normals



# =========================================
# Ray tubes
# =========================================
A_rays = [At_te, Ar_te, At_tm, Ar_tm]                                 # group TE/TM transmitted/reflected coefficients

start = time.perf_counter()

# compute ray-tubes
(triangles, C_ap, A_ap, A_src, dS_src, dS_ap, cos_a, cos_b, Ei_te, Ei_tm, A_rtubes
 ) = tubes.get_rayTubes2(Pk, sk, theta_ts, nk, surfaces, A_rays, ff)   

end = time.perf_counter()
print(f"Calculate ray tubes: {end - start:.4f} sec.")                # timing for ray-tube stage



# =========================================
# Absrobed power calculations
# =========================================
start = time.perf_counter()

refl.get_Pabs(
   A_rtubes, A_ap, A_src, A_src_t,
    Ei_te, Ei_tm, cos_a, cos_b, dS_ap, len(sk)                                                          
)

# A_rtubes, Ak, Ak_src, Ak_src_t, Ei_te, Ei_tm, cos_ap, cos_bp, dLk_ap, nrays_refl

end = time.perf_counter()
print(f"Calculate reflections: {end - start:.4f} sec.")              # timing for absorption/reflection stage


# import input as I
# import numpy as np
# import pyvista as pv
# import mesh
# import plots
# import reflections as refl
# import rayTracing as rt
# import gain as gn
# import rayTubes as tubes
# import readFile as rdFl
# import time
# from numpy import linspace, array, sin, cos

# start = time.perf_counter()

# ############### Input import ##################
# k0 = I.k0                                  
# wv = I.wv
# D = I.D   
# p = I.p
# Nx = I.Nx
# Ny = I.Ny
# bodies = I.bodies
# typeSurface = I.typeSurface
# ################ end input import###########

# surfaces = []
# surfaces = mesh.create_surfaces()
# N_sections = I.nSurfaces 
# Nrays = I.Nrays

# Pk = []
# ray_ids = []
# idx_intersected_faces = []
# nk = []
# sk = []
# ray_lengths = []
# T_tot = []
# e = []
# r_tes = []
# t_tes = []
# r_tms = []
# t_tms = []
# tandels = []
# ndiel = []
# theta_ts = []
# gains = np.zeros(Nrays)
# At_te = []
# Ar_te = []
# At_tm = []
# Ar_tm = []
# extra = 1e-6

# #===========================================================================================================
# if I.typeSrc == 'pw':
#     x = linspace(-I.Lx/2 + extra, I.Lx/2 - extra, I.Nx)
#     y = linspace(-I.Ly/2 + extra, I.Ly/2 - extra, I.Ny) 
#     X, Y = np.meshgrid(x, y)
#     Z = np.zeros_like(X)
#     origins = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
#     theta = np.deg2rad(I.theta_pw)
#     phi   = np.deg2rad(I.phi_pw) 
#     k_dir = array([sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])
#     sk0 = np.tile(k_dir, (Nrays, 1))

# elif I.typeSrc == '2D':
#     x = linspace(-I.Lx/2 + extra, I.Lx/2 - extra, Nrays)
#     y = np.zeros(Nrays)
#     z = np.zeros(Nrays)    
#     origins = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
#     theta = linspace(-np.pi, np.pi - 2*np.pi/(Nrays-1), Nrays)
#     phi   = np.ones(Nrays)*np.deg2rad(0) 
#     x_sk = sin(theta)*cos(phi)
#     y_sk = sin(theta)*sin(phi)
#     z_sk = cos(theta)
#     sk0 = np.column_stack([x_sk, y_sk, z_sk])

# else:
#     origins = mesh.fibonacci_sphere_points(Nrays, I.Lx)
#     sk0 = mesh.fibonacci_sphere_points(Nrays, 1)

# if I.plotSurf: plots.plotSurfaces(surfaces, origins, sk0)

# end = time.perf_counter()
# print(f"Create and plot surfaces : {end - start:.4f} sec.")
# #===========================================================================================================

# #===========================================================================================================
# start = time.perf_counter()
# Ex_src = np.zeros(Nrays, dtype=complex)
# Ey_src = np.zeros(Nrays, dtype=complex)
# Ez_src = np.zeros(Nrays, dtype=complex)

# sx = sk0[:, 0]
# sy = sk0[:, 1]
# sz = sk0[:, 2]
# theta = np.arccos(sz)
# phi = np.mod(np.arctan2(sy, sx), 2*np.pi)
# ff = rdFl.FarFieldInterpolator("Sources/wvgd_src_lin.txt")
# # ff = rdFl.FarFieldInterpolator("Sources/dipole_efield_lin.txt")

# Ex_src, Ey_src, Ez_src = ff.get_cartesian_E(theta, phi)


# A_src_t = np.sqrt(np.abs(Ex_src)**2 + np.abs(Ey_src)**2 + np.abs(Ez_src)**2)      #Amplitude of the electric field in the source
#     # A_src = np.ones(Nrays)

# end = time.perf_counter()
# print(f"Get E-field: {end - start:.4f} sec.")
# #===========================================================================================================


# #===========================================================================================================
# start = time.perf_counter()
# ray_ids, Pk, nk, sk, ndiel, tandels, ray_lengths, theta_ts, r_tes, t_tes, r_tms, t_tms, At_te, Ar_te, At_tm, Ar_tm  = rt.DRT(origins, sk0, surfaces )

# end = time.perf_counter()
# print(f"Calculate ray tracing: {end - start:.4f} sec.")

# if I.plotDRT: plots.plotDRT(surfaces, Pk, sk)
# if I.plotNormals: plots.plot_normals(surfaces, np.real(nk), np.real(Pk))
# #===========================================================================================================


# coefs = [At_te, Ar_te, At_tm, Ar_tm]

# start = time.perf_counter()
# [triangles, C_ap, A_ap, A_src, dS_src, dS_ap, cos_a, cos_b, Ei_te, Ei_tm, coefs2] = tubes.get_rayTubes(Pk, sk, theta_ts, nk, surfaces, coefs, ff)
# end = time.perf_counter()
# print(f"Calculate ray tubes: {end - start:.4f} sec.")

# start = time.perf_counter()
# [At_te2, Ar_te2, At_tm2, Ar_tm2] = coefs2
# refl.get_Pabs(At_te2, Ar_te2, At_tm2, Ar_tm2, A_ap, A_src, A_src_t, Ei_te, Ei_tm, cos_a, cos_b, dS_src, dS_ap, len(sk))
# end = time.perf_counter()
# print(f"Calculate reflections: {end - start:.4f} sec.")






