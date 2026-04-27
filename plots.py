import pyvista as pv
import numpy as np
from matplotlib.cm import get_cmap
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#===========================================================================================================
def plotSurfaces(surfaces, origins, sk0, ray_length=1, rays_as_arrows=False, ray_color="black", ray_width=2, origin_color="red", origin_size=10):
    p = pv.Plotter()                                                        # create PyVista plotter
    for i, si in enumerate(surfaces):
        surf = si.surface                                                   # get surface mesh
        color = si.color                                                    # get surface color
        outer = (i == len(surfaces) - 1)                                    # last surface is the outer one

        p.add_mesh(surf, color=color, opacity=0.05 if outer else 0.35, smooth_shading=not outer, show_edges=False)
        if not outer:                                                       # add wireframe only for internal surfaces
            p.add_mesh(surf, color=color, style="wireframe", line_width=1, opacity=0.9)

    if origins is not None and sk0 is not None:
        norms = np.linalg.norm(sk0, axis=1)                                 # compute direction magnitudes
        valid = norms > 0                                                   # keep only non-zero directions
        dirs = np.zeros_like(sk0)                                           # allocate normalized directions
        dirs[valid] = sk0[valid] / norms[valid, None]                       # normalize valid direction vectors

        # plot ray origins
        p.add_points(origins, color=origin_color, point_size=origin_size, render_points_as_spheres=True)                         

        if rays_as_arrows:
            p.add_arrows(origins[valid], dirs[valid], mag=ray_length,
                         color=ray_color)                                   # plot rays as arrows
        else:                                                               
            lines = [pv.Line(o, o + ray_length * d) for o, d in zip(origins[valid], dirs[valid])] # plot ray origins
            if lines:
                rays_mesh = lines[0]                                        # initialize merged ray mesh
                for ln in lines[1:]:
                    rays_mesh = rays_mesh.merge(ln)                         # merge all ray segments
                p.add_mesh(rays_mesh, color=ray_color, line_width=ray_width)  # plot ray lines                                                        
    p.show()                                                                
#===========================================================================================================


#===========================================================================================================
def plotDRT(surfaces, Pk, show_ray_ids=True, show_all=True):
    plotter = pv.Plotter()                                                # Create plotter

    for i, si in enumerate(surfaces[:-1]):                                # Skip outer surface
        color = 'lightgray' if i == 0 else 'lightblue'                    # Surface color
        plotter.add_mesh(si.surface, color=color, opacity=0.5, show_edges=False)  # Plot surface

    ray_origins, ray_labels = [], []                                      # Label data

    for ray_id, ray_points in enumerate(Pk):                              # Loop over rays
        ray_path = np.asarray(ray_points, dtype=float)                    # Ray points

        valid = np.linalg.norm(ray_path, axis=1) > 0                      # Remove empty points
        ray_path = ray_path[valid]                                        # Keep valid points

        if len(ray_path) < 2:                                             # Need at least 2 points
            continue

        short_ray = len(ray_path) == 2                                    # Direct ray only
        if short_ray and not show_all:
            continue                                                      # Skip direct rays

        ray_color = 'silver' if short_ray else 'black'                    # Ray color

        for p0, p1 in zip(ray_path[:-1], ray_path[1:]):                   # Consecutive points
            plotter.add_lines(np.vstack((p0, p1)), color=ray_color, width=2)  # Plot segment

        if show_ray_ids:
            ray_origins.append(ray_path[1])                               # First hit point
            ray_labels.append(str(ray_id))                                # Ray ID

    if show_ray_ids and ray_origins:
        plotter.add_point_labels(np.asarray(ray_origins), ray_labels, point_size=0, font_size=10, text_color="black")  # Plot IDs

    plotter.add_title("Direct RT", font_size=14)                          # Add title
    plot_axes(plotter, np.array([0, 0.2, 0]))                             # Plot axes
    plotter.show()                                                        # Show plot
#===========================================================================================================


#===========================================================================================================
def plot_axes(plotter, origen):
    eje_x = np.array([0.05, 0, 0])
    eje_y = np.array([0, 0.05, 0])
    eje_z = np.array([0, 0, 0.05])
    plotter.add_arrows(origen, eje_x, mag=1, color="red")
    plotter.add_arrows(origen, eje_y, mag=1, color="green")
    plotter.add_arrows(origen, eje_z, mag=1, color="blue")
    plotter.add_point_labels([origen + eje_x], ['X'], point_size=0, font_size=12, text_color='red')
    plotter.add_point_labels([origen + eje_y], ['Y'], point_size=0, font_size=12, text_color='green')
    plotter.add_point_labels([origen + eje_z], ['Z'], point_size=0, font_size=12, text_color='blue')
 #===========================================================================================================


#===========================================================================================================
def plot_normals(surfaces, nk, Pk):
    plotter = pv.Plotter()                                  # Create plotter

    # 1. Plot all surfaces
    for i in range(len(surfaces) - 1):                     # Skip last surface
        mesh = surfaces[i].surface
        plotter.add_mesh(mesh, opacity=0.4, color='lightgray', show_edges=True)  # Plot surface
        if not mesh.point_data.get("Normals"):             # If no normals
            mesh.compute_normals(point_normals=True, inplace=True)  # Compute normals
        normals = mesh.point_normals                       # Surface normals
        points = mesh.points                               # Surface points

    # 2. Plot normals for each ray, excluding first and last points
    for i in range(len(Pk)):
        points = np.array(Pk[i])                           # Ray points
        normals = np.array(nk[i])                         # Ray normals

        if len(points) <= 2:
            continue                                       # No internal points

        internal_points = points[1:-1]                    # Keep internal points
        internal_normals = normals[1:-1]                  # Keep internal normals

        plotter.add_points( internal_points, color='black', point_size=1, render_points_as_spheres=True)          # Plot points

        point_cloud = pv.PolyData(internal_points)        # Create point cloud
        point_cloud['normals'] = internal_normals         # Assign normals
        arrows = point_cloud.glyph( orient='normals', scale=True, factor=0.005)                                   # Create arrows
        plotter.add_mesh(arrows, color='blue')            # Plot arrows
    plotter.show()                                    
#===========================================================================================================


#===========================================================================================================
def plot_ray_tubes(
        ray_origins,            # Source points
        ray_dirs,               # Ray directions
        Pk_ap,                  # Aperture points
        triangles,              # Triangle indices
        surfaces,
        max_ray_length=None,    # Max ray length
        ray_sample_step=0.1,    # Sampling step
        show_source=True,       # Show source mesh
        show_aperture=True,     # Show aperture mesh
):

    Nrays = ray_origins.shape[0]                           # Number of rays
    plotter = pv.Plotter()                                 # Create plotter

    ## Surfaces
    for i in range(len(surfaces) - 1):                     # Skip last surface
        surf = surfaces[i].surface
        plotter.add_mesh(surf, color="lightblue", opacity=0.5, show_edges=False)                 # Plot surface

    ## Rays
    for i in range(Nrays):
        r0 = ray_origins[i]                                # Ray origin
        s  = ray_dirs[i]                                   # Ray direction
        P  = Pk_ap[i]                                      # Aperture point
        t_int = np.linalg.norm(P - r0)                     # Intersection distance

        if max_ray_length is None:
            ts = np.array([0.0, t_int])                    # End points only
        else:
            t_end = min(max_ray_length, t_int)             # Clip length
            ts = np.linspace(0.0, t_end, int(t_end / ray_sample_step) + 2)  # Sample ray

        ray_points = np.vstack([r0, P])                    # Ray points
        ray_line = pv.Line(r0, P)                          # Create spline
        plotter.add_mesh(ray_line, color="black", line_width=2)  # Plot ray


    ## Source: points + triangulation
    if show_source:
        pts_src = pv.PolyData(ray_origins)                 # Source points
        plotter.add_mesh(pts_src, color="green", point_size=8, render_points_as_spheres=True)                                           # Plot source points

        faces_src = []
        for (i, j, k) in triangles:
            faces_src.extend([3, i, j, k])                 # Build faces

        tri_mesh_src = pv.PolyData(ray_origins, faces_src) # Source mesh
        plotter.add_mesh(tri_mesh_src, color="lightgreen", opacity=0.35, show_edges=True, edge_color="darkgreen", line_width=1)         # Plot source mesh

    ## Aperture: points + triangulation
    if show_aperture:
        pts_ap = pv.PolyData(Pk_ap)                        # Aperture points
        plotter.add_mesh(pts_ap, color="red", point_size=8, render_points_as_spheres=True)                              # Plot aperture points

        faces_ap = []
        for (i, j, k) in triangles:
            faces_ap.extend([3, i, j, k])                  # Build faces

        tri_mesh_ap = pv.PolyData(Pk_ap, faces_ap)         # Aperture mesh
        plotter.add_mesh(tri_mesh_ap, color="cyan", opacity=0.35, show_edges=True, edge_color="blue", line_width=1)     # Plot aperture mesh

    plotter.show()                                         # Show figure
#===========================================================================================================