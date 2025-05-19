import pyvista as pv
import numpy as np
from matplotlib.cm import get_cmap


def plotSurfaces(surfaces):
    p = pv.Plotter()    
    for i in range(0, len(surfaces)):
        
        si = surfaces[i]
        if i == len(surfaces)-1:    
            p.add_mesh(si.surface, show_edges=False, opacity=0.0, color='pink', style='wireframe')
        else:
            p.add_mesh(si.surface, opacity=0.8, color = True, style='wireframe')
    p.show()



def plotDRT(surfaces, Pk, sk, show_ray_ids=True, show_dirs=False, dir_scale=0.04, show_all = False):
    plotter = pv.Plotter()
    cmap = get_cmap("viridis")
    
    # Plot surfaces
    for i in range(len(surfaces)-1):
        if i == 0:
            color = 'lightgray'
        else:
            color = 'lightblue'
        surf = surfaces[i]
        mesh = surf.surface
        plotter.add_mesh(mesh, color=color, opacity=0.5, show_edges=False)
    
    ray_origins = []
    ray_labels = []
    for ray_id, rp in enumerate(Pk):
        # ray_points = rp[:-1]
        ray_points = rp
        n_points = len(ray_points)
        if n_points > 1:
            ray_path = np.array(ray_points)
            if len(ray_path) == 2:
                ray_color = 'silver'
                point_color = 'red'
                if not show_all: continue 
            else:
                ray_color = 'black'
                point_color = 'lightblue'
            for i in range(n_points - 1):
                segment = np.array([ray_path[i], ray_path[i+1]])
                plotter.add_lines(segment, color=ray_color, width=2)
            # for pt in ray_path:
            #     plotter.add_mesh(pv.Sphere(radius=0.0001, center=pt), color=point_color)
            
            if show_dirs:
                for pt, dir_vec in zip(ray_path, sk[Pk.index(ray_points)]):  # synchronize points and directions
                    dir_vec = np.array(dir_vec)
                    if np.linalg.norm(dir_vec) > 0:
                        arrow = pv.Arrow(start=pt, direction=dir_vec, scale=dir_scale)
                        plotter.add_mesh(arrow, color='black')

            if show_ray_ids:
                ray_origins.append(ray_path[1])
                ray_labels.append(str(ray_id))

    if show_ray_ids and ray_origins:
        ray_origins = np.array(ray_origins)
        plotter.add_point_labels(ray_origins, ray_labels, point_size=0, font_size=10, text_color="black")   

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
    if 0:
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