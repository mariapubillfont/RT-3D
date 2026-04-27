import input as I
import rayTracing as rt
import numpy as np
import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp


# =========================================
# Surface class
# =========================================
class Surface:
    def __init__(self, surface, er_in, tand_in, isArray, isAperturePlane, isLastSurface, isFirstIx, color):
        
        mesh_n = surface.compute_normals(
            point_normals= False,   #True,
            cell_normals=True,
            consistent_normals=True,
            auto_orient_normals=True,
            split_vertices=False,
            inplace=False
        )

        self.surface = surface                              # pyvista PolyData
        self.faces = surface.faces                          # faces of the surface
        self.nodes = surface.points                         # nodes of the surface
        self.cell_normals = mesh_n.cell_data["Normals"]
        self.faces_tri = mesh_n.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
        self.er_in = er_in                                  # permittivity "below" (closer to the array) the surface
        self.tand_in = tand_in                              # loss tangent  inside the surface
        self.isArray = isArray                          # true if the surface is the array
        self.isAperturePlane = isAperturePlane          # true if the surface is the aperture plane
        self.isLastSurface = isLastSurface              # true if the surface is the last surface of the radome
        self.isFirstIx = isFirstIx
        self.color = color




# =========================================
# Functions to create the meshes and surfaces
# =========================================

#===========================================================================================================
def create_surfaces():
    surfaces = []
    bodies = I.bodies
    extra_x = 1e-3                                           # small expansion of source plane in x
    extra_y = 1e-3                                           # small expansion of source plane in y
    Lx = I.Lx
    Ly = I.Ly
    bckg_er = I.bckg_er
    bckg_tand = I.bckg_tand

    #------ MESH FOR THE SOURCE -------
    if I.typeSrc == 'pw':
        corners_array = np.array([                                   # rectangular source plane
            [-(Lx / 2 + extra_x), -(Ly / 2 + extra_y), 0.0],
            [-(Lx / 2 + extra_x), +(Ly / 2 + extra_y), 0.0],
            [+(Lx / 2 + extra_x), +(Ly / 2 + extra_y), 0.0],
            [+(Lx / 2 + extra_x), -(Ly / 2 + extra_y), 0.0]])                                                  
        src_mesh = create_rectangular_surf(corners_array)           # source mesh for plane-wave excitation
    else:
        src_mesh = create_full_sphere([0, 0, 0], I.Lx)              # spherical source surface for antenna excitation


    #surface(surface, er_in, tand_in, isArray, isAperturePlane, isLastSurface, isFirstIx, color)
    src_surface = Surface(src_mesh, bckg_er, bckg_tand, True, False, False, True, "white")
    surfaces.append(src_surface)                                    # add the source surface to the array of "surfaces"

    #------ MESH FOR THE OTHER STRUCTURES -------
    for ii, body in enumerate(bodies):
        if body.type == 'cylinder':
            mesh_obj = create_full_cylinder(body.radius, body.height, body.center, body.axis)                                                

        elif body.type == 'box':
            mesh_obj = create_full_box(body.center, body.axis)                                                

        elif body.type == 'ellipse':
            mesh_obj = create_elliptical_cylinder(body.center, [body.a, body.b, body.h])                                                

        else:
            raise ValueError(f"Unsupported surface type: {body.type}")

        er_in = body.er_in                                      # material of current body
        tand_in = body.tand_in

       
        #surface(surface, er_in, tand_in, isArray, isAperturePlane, isLastSurface, isFirstIx, color)
        surf = Surface(mesh_obj, er_in, tand_in, False, False,  False, False, bodies[0].color)
        surfaces.append(surf)                                   # add the bodi_i surface to the array of "surfaces"

    #------ MESH FOR THE APERTURE ------
    out_mesh = create_full_sphere([0, 0, 0], I.D)               # aperture sphere
    
    #surface(surface, er_in, tand_in, isArray, isAperturePlane, isLastSurface, isFirstIx, color)
    out_surface = Surface(out_mesh, bckg_er, bckg_tand, False, False, True, False, "white")
    surfaces.append(out_surface)                                # add the aperture surface to the array of "surfaces"

    return surfaces
#===========================================================================================================


#===========================================================================================================
def create_rectangular_surf(points):
    gmsh.initialize()                                                          # initialize Gmsh session
    model, occ, mesh = gmsh.model, gmsh.model.occ, gmsh.model.mesh            # shortcuts for model, geometry and mesh

    pts = [occ.addPoint(*points[i], 0.0) for i in range(4)]                   # create the 4 corner points of the rectangle
    lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]         # create edges connecting consecutive points
    loop = occ.addCurveLoop(lines)                                            # create closed boundary loop
    occ.addPlaneSurface([loop])                                               # create planar rectangular surface

    occ.synchronize()                                                         # synchronize CAD kernel with Gmsh model

    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)                          # set minimum mesh size
    gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)                  # set maximum mesh size
    mesh.generate(2)                                                          # generate 2D triangular mesh

    _, nodeCoords, _ = mesh.getNodes()                                        # retrieve node coordinates
    elemType = mesh.getElementType("triangle", 1)                             # get triangle element type
    faceNodes = mesh.getElementFaceNodes(elemType, 3)                         # get triangle connectivity (faces)

    nodes = nodeCoords.reshape(-1, 3)                                         # reshape node array to (N, 3)
    faces = faceNodes.reshape(-1, 3)                                          # reshape face array to (M, 3)

    gmsh.finalize()                                                           # finalize Gmsh session

    ind0 = int(faces.min()) - 1                                               # convert from 1-based to 0-based indexing
    V = nodes[ind0:]                                                          # keep valid subset of nodes
    F = faces - (ind0 + 1)                                                    # shift face indices accordingly
    faces_pv = np.hstack((3*np.ones((F.shape[0], 1)), F)).astype(int)         # build PyVista face format [3, i0, i1, i2]

    surf = pv.PolyData(V, faces=faces_pv)                                     # create PyVista surface mesh
    return surf                                                               # return rectangular surface
#===========================================================================================================


#===========================================================================================================
def create_full_box(center, axis, rotation_deg = 0):
    gmsh.initialize()
    model = gmsh.model
    occ   = model.occ
    mesh  = model.mesh

    cx, cy, cz = center
    dx, dy, dz = axis
    tag = occ.addBox(cx, cy, cz, dx, dy, dz)
    if rotation_deg != 0:
        center_x = cx + dx / 2
        center_y = cy + dy / 2
        center_z = cz + dz / 2
        angle_rad = np.radians(rotation_deg)
        occ.rotate([(3, tag)], center_x, center_y, center_z, 0, 0, 1, angle_rad)
    occ.synchronize()

    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.01)   
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
def create_full_sphere(center, radius):
    gmsh.initialize()
    model = gmsh.model
    occ   = model.occ
    mesh  = model.mesh
    cx, cy, cz = center
    R = radius

    occ.addSphere(cx, cy, cz, R)
    occ.synchronize()

    gmsh.option.setNumber('Mesh.MeshSizeMin', R/20)
    gmsh.option.setNumber('Mesh.MeshSizeMax', R/5)   
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
def sphere_sampling(n_pts, R, center=(0.0, 0.0, 0.0), randomize=False):
    # Generates quasi-uniform points on a sphere
    # 1) Points are evenly spaced in the vertical direction (z-axis)
    # 2) Each point is rotated by a constant angle (golden angle) to spread them around

    GR = (1.0 + np.sqrt(5.0)) / 2.0                     # golden ratio
    golden_angle = 2.0 * np.pi * (1.0 - 1.0/GR)         # golden angle (~137.5 deg)

    if randomize:
        phase = np.random.random() * 2*np.pi            # random rotation to avoid fixed pattern
    else:
        phase = 0.0                                    # deterministic distribution

    i = np.arange(n_pts)                               # point indices

    mu = 1.0 - 2.0 * (i + 0.5) / n_pts                 # uniform sampling of cos(theta)
    theta = np.arccos(np.clip(mu, -1.0, 1.0))          # polar angle
    phi = i * golden_angle + phase                     # azimuthal angle

    x = R * np.sin(theta) * np.cos(phi)                # x coordinate
    y = R * np.sin(theta) * np.sin(phi)                # y coordinate
    z = R * np.cos(theta)                             # z coordinate

    pts = np.column_stack([x, y, z]) + np.array(center)[None, :]  # shift to desired center
    return pts
#===========================================================================================================



#===========================================================================================================
def create_full_cylinder(radius, height, center, axis):
    gmsh.initialize()                                                           # Initialize Gmsh
    model = gmsh.model                                                          # Choose the model and OCC kernel
    occ   = model.occ
    mesh  = model.mesh

    cx, cy, cz = center                                                         # Unpack center
    ax, ay, az = axis

    occ.addCylinder(cx, cy, cz, ax*height, ay*height, az*height, radius)
    occ.synchronize()
    
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)                            # Set the meshing algorithm and generate the mesh
    gmsh.option.setNumber('Mesh.MeshSizeMax', I.meshMaxSize)   
    mesh.generate(2)                                                            # Generate a 2D mesh

    nodeTags, nodeCoords, _ = model.mesh.getNodes()                             # get the nodes
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    faceNodes = gmsh.model.mesh.getElementFaceNodes(elementType, 3)             # get the faces
    nodes = nodeCoords.reshape(-1, 3)                                           # reshape vector size to [nNodes, 3]
    faces = np.reshape(faceNodes, (-1, 3))                                      # reshape vector size to [nFaces, 3]

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
def create_elliptical_cylinder(center, axis, mesh_size = I.meshMaxSize):
    gmsh.initialize()
    model = gmsh.model
    occ = model.occ
    mesh = model.mesh
    cx, cy, cz = center
    a, b, h = axis
    ellipse_center_y = cy - h / 2

    tag = occ.addEllipse(cx, ellipse_center_y, cz, a, b, zAxis=[0, 1, 0],     # normal along y → ellipse in XZ
        xAxis=[1, 0, 0])
    wire = occ.addWire([tag])
    surface = occ.addPlaneSurface([wire])
    ov = occ.extrude([(2, surface)], 0, h, 0)  # (dim=2, tag), along y
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    mesh.generate(2)
    nodeTags, nodeCoords, _ = model.mesh.getNodes()
    elementType = model.mesh.getElementType("triangle", 1)
    faceNodes = model.mesh.getElementFaceNodes(elementType, 3)
    nodes = nodeCoords.reshape(-1, 3)
    faces = faceNodes.reshape(-1, 3)
    gmsh.finalize()
    ind0 = int(np.min(faces)) - 1
    V = nodes[ind0:, :]
    F = faces - (ind0 + 1)
    faces1 = np.hstack((3 * np.ones((F.shape[0], 1)), F)).astype(int)
    surf = pv.PolyData(V, faces=faces1)
    surf.flip_normals()  # <--- Añade esta línea
    return surf
#===========================================================================================================