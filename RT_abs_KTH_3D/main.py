
import input as I
import Gmsh_examples as ex
import input as I
import numpy as np
import pyvista as pv
import mesh
import plots
import rayTracing as rt


############### Input import ##################
k0 = I.k0                                  
wv = I.wv
D = I.D   
p = I.p
Nx = I.Nx
Ny = I.Ny
bodies = I.bodies
typeSurface = I.typeSurface
################ end input import###########




surfaces = []
surfaces = mesh.create_surfaces()
if I.plotSurf == 1:
    plots.plotSurfaces(surfaces)



N_sections = I.nSurfaces 
N_rays = I.Nrays

# Pk = np.zeros([N_sections+1, N_rays, 3])
# idx_intersected_faces = np.zeros([N_rays, N_sections])
# nk = np.zeros([N_sections+1, N_rays, 3])
# sk = np.zeros([N_sections+1, N_rays, 3])
# path_length = np.zeros((N_rays,1))
Pk = []
idx_intersected_faces = []
nk = []
sk = []
path_length = []
T_tot = []
e = []


origins = np.zeros([N_rays, 3])
sk0 = rt.fibonacci_sphere(N_rays)

Pk, idx_intersected_faces, path_length, nk, sk,  = rt.DRT(origins, Nx, Ny, sk0, surfaces )


plots.plotDRT(surfaces, Pk, N_sections, sk)
# ray_length = 0.5  # longitud de los rayos

# origin = np.array([0, 0, 0])




# plotter = pv.Plotter()

# # Plot surfaces
# for surf in surfaces:
#     plotter.add_mesh(surf.surface, color='lightgray', opacity=0.5, show_edges=True)

# N_used_rays = None
# direction = 'DRT'
# show_dirs = True
# dir_scale = 0.01
# # Determine how many rays to use
# if N_used_rays is None:
#     N_used_rays = Pk.shape[1]

# # Plot each ray
# for i in range(N_used_rays):
#     # Extract the polyline of the ray (from surface 0 to N_sections)
#     ray_path = Pk[:N_sections+1, i, :]
#     n_points = ray_path.shape[0]
#     connectivity = np.hstack([[n_points], np.arange(n_points)])
#     ray_line = pv.PolyData()
#     ray_line.points = ray_path
#     ray_line.lines = connectivity
#     plotter.add_mesh(ray_line, color='red', line_width=2)
#     # if show_dirs and sk is not None:
#     #     for k in range(N_sections):
#     #         # Arrow base = position
#     #         p_start = Pk[k, i, :]
#     #         direction_vec = sk[k, i, :]
#     #         arrow = pv.Arrow(start=p_start, direction=direction_vec, scale=dir_scale)
#     #         plotter.add_mesh(arrow, color='blue')

# plotter.add_title(f"{direction} Ray Paths", font_size=14)
# plotter.show()


# ray_length = 0.2

# plotter = pv.Plotter()
# for s in sk0:
#     end = origin + s * ray_length
#     plotter.add_mesh(pv.Line(origin, end), color='black')

# plotter.add_mesh(pv.Sphere(radius=0.005, center=origin), color='black')
# plotter.add_mesh(surfaces[0].surface, opacity=0.9, color = True)
# # plotter.show_grid()
# plotter.show()

# 3. Crear las líneas de los rayos
# lines = []
# for dir_vec in sk0:
#     end_point = origins[0] + dir_vec * ray_length
#     line = pv.Line(origins[0], end_point)
#     lines.append(line)

# 4. Unir todo en una sola malla
# rays = lines[0]
# for line in lines[1:]:
#     rays = rays.merge(line)

# # 5. Visualizar
# plotter = pv.Plotter()



# for i in range(0, len(surfaces)):
#     si = surfaces[i]
#     plotter.add_mesh(si.surface, opacity=1, color = True)


# plotter.add_mesh(rays, color="red", line_width=2)
# plotter.add_mesh(pv.Sphere(radius=0.01, center=origins[0]), color="blue", name="origin")
# plotter.show_grid()
# plotter.show()

# Pk, idx_intersected_faces, path_length, nk, sk, T_tot, e = rt.DRT()




