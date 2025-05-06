import pyvista as pv
import numpy as np
from matplotlib.cm import get_cmap


def plotSurfaces(surfaces):
    p = pv.Plotter()    
    for i in range(0, len(surfaces)):
        si = surfaces[i]
        if i == len(surfaces)-1:    
            p.add_mesh(si.surface, show_edges=False, opacity=0.2, color='pink', style='wireframe')
        else:
            p.add_mesh(si.surface, opacity=0.8, color = True)
    # p.show_grid()
    p.show()



def plotDRT(surfaces, Pk, N_sections, sk, N_used_rays=None, show_dirs=True, dir_scale=0.03):
    """
    Plot the ray paths (and optionally the ray direction vectors) along the given surfaces.

    Parameters:
    - surfaces: list of pyvista.PolyData surfaces
    - Pk: (N_sections+1, N_rays, 3) ray positions at each interface
    - N_sections: int, number of surfaces/layers
    - sk: (N_sections+1, N_rays, 3), optional ray directions
    - N_used_rays: int, optional number of rays to plot
    - direction: 'DRT' or 'IRT' or label
    - show_dirs: bool, whether to draw arrows for directions
    - dir_scale: float, scale of direction arrows
    """
    plotter = pv.Plotter()
    cmap = get_cmap("viridis")

    
    # Plot surfaces
    for i in range(len(surfaces)-1):
        surf = surfaces[i]
        plotter.add_mesh(surf.surface, color='lightgray', opacity=0.5, show_edges=True)

    # Determine how many rays to use
    if N_used_rays is None:
        N_used_rays = Pk.shape[1]

    # Plot each ray
    for i in range(N_used_rays):
        # Extract the polyline of the ray (from surface 0 to N_sections)
        ray_path = Pk[:N_sections+1, i, :]
        n_points = ray_path.shape[0]
        connectivity = np.hstack([[n_points], np.arange(n_points)])
        ray_line = pv.PolyData()
        ray_line.points = ray_path
        ray_line.lines = connectivity
        plotter.add_mesh(ray_line, color='black', line_width=2)

        if show_dirs: #and sk is not None:
            for k in range(N_sections):
                # Arrow base = position
                color = cmap(k / (N_sections - 1))
                p_start = Pk[k, i, :]
                direction_vec = sk[k, i, :]
                arrow = pv.Arrow(start=p_start, direction=direction_vec, scale=dir_scale)
                plotter.add_mesh(arrow, color=color)

    plotter.add_points(Pk[1], color='black', point_size=5, render_points_as_spheres=True)

    plotter.add_title("Direct RT", font_size=14)
    plotter.show()





def plot_normals(surface, points, interp_vals, surface_nodes, normals_nodes):

    """
    Visualizes:
    - the original surface mesh
    - the normals at each surface node (in red)
    - the interpolated normals at specific points (in blue)
    
    Parameters:
    - surface: pyvista.PolyData → the surface mesh
    - points: (N, 3) → coordinates of points where normals were interpolated
    - interp_vals: (N, 3) → interpolated normal vectors at those points
    - surface_nodes: (M, 3) → coordinates of mesh nodes
    - normals_nodes: (M, 3) → normal vectors at each mesh node
    - scale_node: float → arrow scale for surface node normals
    - scale_interp: float → arrow scale for interpolated normals
    """
    plotter = pv.Plotter()

    # 1. Plot the surface mesh
    plotter.add_mesh(surface, color='lightgray', opacity=0.5, show_edges=True)

    # # 2. Plot normals at the surface nodes (in red)
    if 1:
        surface_pc = pv.PolyData(surface_nodes)
        surface_pc['normals'] = normals_nodes
        arrows_nodes = surface_pc.glyph(orient='normals', scale=True, factor=0.05)
        plotter.add_mesh(arrows_nodes, color='red', label='Nodal Normals')

    # 3. Plot interpolated normals at given points (in blue)
    interp_pc = pv.PolyData(points)
    interp_pc['normals'] = interp_vals
    arrows_interp = interp_pc.glyph(orient='normals', scale=True, factor=0.05)
    plotter.add_mesh(arrows_interp, color='blue', label='Interpolated Normals')

    # 4. Plot the points where normals were interpolated (in black)
    plotter.add_points(points, color='black', point_size=5, render_points_as_spheres=True)

    plotter.add_legend()
    plotter.show()