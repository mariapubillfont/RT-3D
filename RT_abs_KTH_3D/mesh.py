import input as I
import rayTracing as rt
import numpy as np
import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp

############################################## FUNCTIONS ###################################################
#===========================================================================================================
def loop_revolution_surf(surf, params, surfaces, ii):
    type_surface, type_surf_n_params, surf_params, nSurfaces, MLthick, er = params
    # for ii in range(I.nSurfaces):
    surf = create_revolution_surf(type_surface[ii], MLthick[ii],surf_params[ii*type_surf_n_params[ii]:ii*type_surf_n_params[ii]+type_surf_n_params[ii]])
    if ii == (nSurfaces-1): isLastSurf = True
    else: isLastSurf = False
    surf = rt.Surface(surf, er[ii], er[ii+1], False, False, isLastSurf)
    # surfaces.append(surf)
    
    return surf

# For parallelization
def slice_data(data, nprocs):
    aver, res = divmod(len(data), nprocs)
    nums = []
    for proc in range(nprocs):
        if proc < res:
            nums.append(aver + 1)
        else:
            nums.append(aver)
    count = 0
    slices = []
    for proc in range(nprocs):
        slices.append(data[count: count+nums[proc]])
        count += nums[proc]
    return slices


def create_surfaces():
    # type_surface, type_surf_n_params, surf_params, nSurfaces, MLthick, er = params
    surfaces = []
    nSurfaces = I.nSurfaces
    bodies = I.bodies
    er = I.er
    tand = I.tand
    extra = 0.005
        
    corners_array = np.array([np.array([-(I.Lx/2+extra),-(I.Ly/2+extra), 0.]),
                                np.array([-(I.Lx/2+extra),+(I.Ly/2+extra), 0.]),
                                np.array([+(I.Lx/2+extra),+(I.Ly/2+extra), 0.]),
                                np.array([+(I.Lx/2+extra),-(I.Ly/2+extra), 0.])])
    surf = create_rectangular_surf(corners_array)
    surf = rt.Surface(surf, 1., er[0], 0, 0, True, False, False, True)
    surfaces.append(surf)

    # nprocs = min(mp.cpu_count(), I.nSurfaces)
    # inp_lists = slice_data(range(I.nSurfaces), nprocs)
    # pool = mp.Pool(processes=nprocs)
    # multi_result = [pool.apply_async(loop_revolution_surf, (surf, params, surfaces)) for inp in inp_lists]
    # multi_result = [pool.apply_async(loop_revolution_surf, (surf, params, surfaces, ii)) for ii in range(I.nSurfaces)]
    # result = [x for p in multi_result for x in p.get()]
    # result = [p.get() for p in multi_result]
    # surfaces.extend([p.get() for p in multi_result])
    
    for ii in range(nSurfaces-1):
        match I.typeSurface[ii]:
            case 'cylinder':
                surf = create_full_cylinder(bodies.r, bodies.h, bodies.center, bodies.ax)

            case 'box':
                surf = create_full_box(bodies.center, bodies.ax)

            # case 'ellipse':
            #     surf = Ellipse(variables['center1'], variables['a'], variables['b'], variables['c'])
        

        if ii == (nSurfaces-1): isLastSurf = True
        else: isLastSurf = False
        surf = rt.Surface(surf, er[ii], er[ii+1],  tand[ii], tand[ii+1], False, False, False, True)
        surfaces.append(surf)
    
    surf = create_full_sphere()
    surf = rt.Surface(surf, 1, 1,  1, 1, False, False, True, True)
    surfaces.append(surf)
    return surfaces

#===========================================================================================================

#===========================================================================================================

def curve_function(t, type_surf, MLthick, surf_params):
    match type_surf:
        case 'conic':
            if MLthick == 0:
                x = t[t<surf_params[3]]
                y = np.zeros_like(t[t<surf_params[3]])
                z = I.conic_function([surf_params[0], surf_params[1], surf_params[2]])(t[t<surf_params[3]])
            else:
                x = I.matchingLayer_x(I.conic_function([surf_params[0], surf_params[1], surf_params[2]]), MLthick)(t[t<surf_params[3]])
                y = np.zeros_like(t[t<surf_params[3]])
                z = I.matchingLayer_z(I.conic_function([surf_params[0], surf_params[1], surf_params[2]]), MLthick)(t[t<surf_params[3]])
        case 'semicirc':
            x = t
            y = np.zeros_like(t)
            z = I.semicirc_function([surf_params[0], surf_params[1], surf_params[2]])(t)
        case 'line':
            x = t
            y = np.zeros_like(t)
            z = I.line_function([surf_params[0], surf_params[1]])(t)
    return x, y, z

#===========================================================================================================



#===========================================================================================================
def create_full_cylinder2(radius, height, center, axis):
    gmsh.initialize()
    model = gmsh.model
    occ   = model.occ
    mesh  = model.mesh
    cx, cy, cz = center
    ax, ay, az = axis

    
    x_values = np.linspace(0, radius, I.num_points)
    y_values = np.zeros(I.num_points)
    z_values = np.ones(I.num_points)*height      # fer-ho de forma matricial per eliminar les columnes amb valors NaN!!!!!!!!

    point_ids = np.zeros_like(x_values)
    for i in range(len(point_ids)):
        point_id = occ.addPoint(x_values[i], y_values[i], z_values[i], meshSize=0.1)
        point_ids[i] = point_id

    l1 = occ.addSpline(point_ids)
    occ.synchronize()   

    surf1 = occ.revolve([(1, l1)], 0, 0, 0, 0, 0, 1, 2*np.pi)
    occ.synchronize()

    # occ.addCylinder(cx, cy, cz, ax*height, ay*height, az*height, radius)
    # occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
    gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)   
    mesh.generate(2)

    nodeTags, nodeCoords, _ = model.mesh.getNodes()                         # get the nodes
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)         # get the faces
    nodes = nodeCoords.reshape(-1, 3)                                       # reshape vector size to [nNodes, 3]
    faces = np.reshape(faceNodes, (-1, 3))                                  # reshape vector size to [nFaces, 3]
    occ.synchronize()
    gmsh.finalize()

    ind0 = int(np.min(faces))-1                                             # shift from 1- to 0-based indexing
    V = nodes[ind0:,:]                                                      # discarting nodes with index < ind0
    F = faces-(ind0+1)                                                      # Adjusts all the indices in faces by subtracting (ind0 + 1), from 1- to 0-based indexing.                    
    faces1 = np.hstack((3*np.ones((F.shape[0],1)),F)).astype(int)           # for pyvista each triangular cell is: [3, i0, i1, i2]    
    surf = pv.PolyData(V,faces=faces1) #,n_faces=faces1.shape[0]
    return surf
#===========================================================================================================





#===========================================================================================================
def create_full_cylinder(radius, height, center, axis):
# Creates a full cylinder (including top, bottom, and curved surface) with a 2D triangular mesh
# and returns a PyVista PolyData that shows all triangles

    # Initialize Gmsh
    gmsh.initialize()

    # Choose the model and OCC kernel
    model = gmsh.model
    occ   = model.occ
    mesh  = model.mesh

    # Unpack center
    cx, cy, cz = center
    ax, ay, az = axis

    # Create a solid cylinder using addCylinder(x0, y0, z0, dx, dy, dz, radius)
    # (x0, y0, z0) is the bottom center, (dx, dy, dz) is the direction vector for the axis
    occ.addCylinder(cx, cy, cz, ax*height, ay*height, az*height, radius)

    # Synchronize geometry
    occ.synchronize()

    # Set the meshing algorithm and generate the mesh
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
    gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)   

    # Generate a 2D mesh
    mesh.generate(2)

    # Get node coordinates
    nodeTags, nodeCoords, _ = model.mesh.getNodes()                         # get the nodes
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)         # get the faces

    nodes = nodeCoords.reshape(-1, 3)                                       # reshape vector size to [nNodes, 3]
    faces = np.reshape(faceNodes, (-1, 3))                                  # reshape vector size to [nFaces, 3]

    occ.synchronize()
    gmsh.finalize()

    # To create the mesh in pyvista
    ind0 = int(np.min(faces))-1                                             # shift from 1- to 0-based indexing
    V = nodes[ind0:,:]                                                      # discarting nodes with index < ind0
    # V[:,0] = 0.7*V[:,0]                                                   # There is  compression in the x direction
    F = faces-(ind0+1)                                                      # Adjusts all the indices in faces by subtracting (ind0 + 1), from 1- to 0-based indexing.                    
    faces1 = np.hstack((3*np.ones((F.shape[0],1)),F)).astype(int)           # for pyvista each triangular cell is: [3, i0, i1, i2]    
    # faces_flat = faces1.flatten()
    surf = pv.PolyData(V,faces=faces1) #,n_faces=faces1.shape[0]
    return surf
#===========================================================================================================


#===========================================================================================================
def create_full_box(center, axis):
    gmsh.initialize()
    model = gmsh.model
    occ   = model.occ
    mesh  = model.mesh

    cx, cy, cz = center
    dx, dy, dz = axis

    occ.addBox(cx, cy, cz, dx, dy, dz)
    occ.synchronize()

    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
    gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)   

    mesh.generate(2)

    nodeTags, nodeCoords, _ = model.mesh.getNodes()                         # get the nodes
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)         # get the faces
    nodes = nodeCoords.reshape(-1, 3)                                       # reshape vector size to [nNodes, 3]
    faces = np.reshape(faceNodes, (-1, 3))                                  # reshape vector size to [nFaces, 3]

    occ.synchronize()
    gmsh.finalize()

    ind0 = int(np.min(faces))-1                                             # shift from 1- to 0-based indexing
    V = nodes[ind0:,:]                                                      # discarting nodes with index < ind0
    F = faces-(ind0+1)                                                      # Adjusts all the indices in faces by subtracting (ind0 + 1), from 1- to 0-based indexing.                    
    faces1 = np.hstack((3*np.ones((F.shape[0],1)),F)).astype(int)           # for pyvista each triangular cell is: [3, i0, i1, i2]    
    surf = pv.PolyData(V,faces=faces1) #,n_faces=faces1.shape[0]
    return surf
#===========================================================================================================



#===========================================================================================================
def create_full_sphere():
    gmsh.initialize()
    model = gmsh.model
    occ   = model.occ
    mesh  = model.mesh

    cx, cy, cz = [0, 0,0]

    occ.addSphere(cx, cy, cz, I.D)
    occ.synchronize()

    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
    gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)   
    mesh.generate(2)

    # Get node coordinates
    nodeTags, nodeCoords, _ = model.mesh.getNodes()                         # get the nodes
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)         # get the faces
    nodes = nodeCoords.reshape(-1, 3)                                       # reshape vector size to [nNodes, 3]
    faces = np.reshape(faceNodes, (-1, 3))                                  # reshape vector size to [nFaces, 3]

    occ.synchronize()
    gmsh.finalize()

    ind0 = int(np.min(faces))-1                                             # shift from 1- to 0-based indexing
    V = nodes[ind0:,:]                                                      # discarting nodes with index < ind0
    F = faces-(ind0+1)                                                      # Adjusts all the indices in faces by subtracting (ind0 + 1), from 1- to 0-based indexing.                    
    faces1 = np.hstack((3*np.ones((F.shape[0],1)),F)).astype(int)           # for pyvista each triangular cell is: [3, i0, i1, i2]    
    surf = pv.PolyData(V,faces=faces1) #,n_faces=faces1.shape[0]
    return surf
#===========================================================================================================



#===========================================================================================================

def create_revolution_surf(type_surface, MLthick, surf_params):
    
    gmsh.initialize()

    # Choose kernel
    model = gmsh.model
    occ = model.occ
    mesh = model.mesh

    # Create curve points
    t = t = np.linspace(0, surf_params[-1], I.num_points)
    x_values, y_values, z_values = curve_function(t, type_surface, MLthick, surf_params)      # fer-ho de forma matricial per eliminar les columnes amb valors NaN!!!!!!!!

    # Add points to Gmsh
    point_ids = np.zeros_like(x_values)
    for i in range(len(point_ids)):
        point_id = occ.addPoint(x_values[i], y_values[i], z_values[i], meshSize=0.1)
        point_ids[i] = point_id

    # Create a spline curve through the points
    l1 = occ.addSpline(point_ids)
    # Create copies and rotate
    l2 = occ.copy([(1,l1)])
    l3 = occ.copy([(1,l1)])

    # Rotate the copy
    occ.rotate(l2, 0, 0, 0, 0, 0, 1, 2*np.pi/3)
    occ.rotate(l3, 0, 0, 0, 0, 0, 1, 4*np.pi/3)

    occ.synchronize()    

    # Create the surface of revolution
    surf1 = occ.revolve([(1, l1)], 0, 0, 0, 0, 0, 1, 2*np.pi/3)
    surf2 = occ.revolve(l2, 0, 0, 0, 0, 0, 1, 2*np.pi/3)
    surf3 = occ.revolve(l3, 0, 0, 0, 0, 0, 1, 2*np.pi/3)

    # Join the surfaces
    surf4 = occ.fragment(surf1, surf2)
    surf_tag = occ.fragment(surf4[0], surf3)

    # mesh.setOutwardOrientation(surf_tag)

    occ.synchronize()
    
        
    # Set the meshing algorithm and generate the mesh
    if 0: # To generate quadrangles
        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
        gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
        gmsh.option.setNumber('Mesh.ElementOrder', 2)    
    if 1:
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
        gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)   
        
    mesh.generate(2)
    
    # print(gmsh.model.getEntities())

    # To get nodes, faces and edges
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)

    nodes = np.reshape(nodeCoords, (-1, 3))
    faces = np.reshape(faceNodes, (-1, 3))
    if 0:
        print("Nodes:")
        print(nodes)
        print("Faces:")
        print(faces)

    occ.synchronize()
    # gmsh.fltk.run()       # visualize in gmsh GUI

    # Save the mesh to a file (optional)
    #gmsh.write("surface_of_revolution.msh")

    gmsh.finalize()

    # To create the mesh in pyvista
    ind0 = int(np.min(faces))-1
    V = nodes[ind0:,:]
    # V[:,0] = 0.7*V[:,0]  # There is  compression in the x direction
    F = faces-(ind0+1)
    faces1 = np.hstack((3*np.ones((F.shape[0],1)),F)).astype(int)
    # faces_flat = faces1.flatten()
    surf = pv.PolyData(V,faces=faces1) #,n_faces=faces1.shape[0]

    
    return surf

#===========================================================================================================

#===========================================================================================================

def create_rectangular_surf(points):
    
    gmsh.initialize()

    # Choose kernel
    model = gmsh.model
    occ = model.occ
    mesh = model.mesh

    # Create points
    pt1 = occ.addPoint(points[0,0], points[0,1], points[0,2], 0., -1)
    pt2 = occ.addPoint(points[1,0], points[1,1], points[1,2], 0., -1)
    pt3 = occ.addPoint(points[2,0], points[2,1], points[2,2], 0., -1)
    pt4 = occ.addPoint(points[3,0], points[3,1], points[3,2], 0., -1)
    # Create lines
    lines = np.array([occ.addLine(pt1, pt2, -1), occ.addLine(pt2, pt3, -1), occ.addLine(pt3, pt4, -1), occ.addLine(pt4, pt1, -1)])
    # Create wire rectangle
    wire_rectangle = occ.addCurveLoop(lines, -1)
    # Create surface rectangle
    rectangle = occ.addPlaneSurface(np.array([wire_rectangle]), -1)

    # mesh.setOutwardOrientation(rectangle)

    occ.synchronize()

    # Set the meshing algorithm and generate the mesh
    if 0: # To generate quadrangles
        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
        gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
        gmsh.option.setNumber('Mesh.ElementOrder', 2)    
    if 1:
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
        gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)  

    mesh.generate(2)

    # To get nodes, faces and edges
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)

    nodes = np.reshape(nodeCoords, (-1, 3))
    faces = np.reshape(faceNodes, (-1, 3))


    occ.synchronize()
    gmsh.finalize()

    # To create the mesh in pyvista
    ind0 = int(np.min(faces))-1
    V = nodes[ind0:,:]
    F = faces-(ind0+1)
    faces1 = np.hstack((3*np.ones((F.shape[0],1)),F)).astype(int)
    surf = pv.PolyData(V,faces=faces1) #,n_faces=faces1.shape[0]

    # p = pv.Plotter()
    # p.add_mesh(surf, show_edges=True, opacity=0.7, lighting=False)
    # p.show_grid()
    # p.show()

    return surf

#===========================================================================================================

#===========================================================================================================

def aperture_plane_points(angle, L):
    
    # Corners calculation
    theta_0 = angle[0]
    phi_0 = angle[1]
    v = np.array([np.sin(theta_0)*np.cos(phi_0), np.sin(theta_0)*np.sin(phi_0), np.cos(theta_0)])
    A = I.distance_ap*v
    u = np.random.randn(3)
    u -= np.dot(u,v)*v
    u /= np.linalg.norm(u)
    w = np.cross(v,u)
    B = A - L*u - L*w
    C = A + L*u - L*w
    D = A + L*u + L*w
    E = A - L*u + L*w

    # Rays' origins calculations
    array_x = np.linspace(0, 2*L, int(np.sqrt(I.N_rrt)))
    array_y = np.linspace(0, 2*L, int(np.sqrt(I.N_rrt)))
    array2D_x, array2D_y = np.meshgrid(array_x, array_y)
    meshgrid = np.vstack([array2D_x.flatten(), array2D_y.flatten()])
    base = np.hstack([w.reshape(-1,1), u.reshape(-1,1)])
    extended_B = np.tile(B.reshape(3,1),(1,I.N_rrt))
    points = extended_B + np.dot(base,meshgrid)

    return np.array([B, C, D, E]), points, -v

#===========================================================================================================



####################################### EXECUTE FOR DEBUGGING ##############################################
#===========================================================================================================

if __name__ == "__main__":  # ML not implemented
    
    if 1: # To plot the mesh using pyvista
        surf = create_revolution_surf(I.type_surface, I.surfaces_parameters)
        # surf.plot_normals(mag=0.1, show_mesh=True, flip=True,show_edges=True)
        p = pv.Plotter()
        p.add_mesh(surf, show_edges=True)
        p.show_grid()
        p.save_graphic(I.newpath_results + '/revolution_surf_mesh.pdf') 
        p.show()
        # p = pvqt.BackgroundPlotter()
        # p.add_mesh(surf, show_edges=True)
        # p.show_grid()
        # p.save_graphic(I.newpath_results + '/revolution_surf_mesh.pdf')
        # p.show()
        # p.app.exec_()

    
    if 0: # To plot the mesh using pyvista
        surf = create_rectangular_surf(I.Lx, I.Ly)
        p = pv.Plotter()
        p.add_mesh(surf, show_edges=True)
        p.show_grid()
        p.save_graphic(I.newpath_results + '/rectangular_surf_mesh.pdf') 
        p.show() 
        # p = pvqt.BackgroundPlotter()
        # p.add_mesh(surf, show_edges=True)
        # p.show_grid()
        # p.save_graphic(I.newpath_results + '/revolution_surf_mesh.pdf')
        # p.show()
        # p.app.exec_()

 