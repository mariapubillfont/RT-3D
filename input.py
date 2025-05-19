import re
import numpy as np

class Cylinder:
    def __init__(self, radius, height, axis, center, color):
        self.r = radius
        self.h = height
        self.ax = axis
        self.center = center
        self.color = color

class Box:
    def __init__(self, center, axis):
        self.center = center
        self.ax = axis

class Ellipse:
    def __init__(self, center, a, b, c):
        self.center = center
        self.a = a
        self.b = b
        self.c =  c


variables = {}
with open('In_files/inputBox.txt', 'r') as file:
# with open('In_files/inPlWv.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        try:
            # Convert to float or array of floats if possible
            values = [float(x) for x in value.split()]
            variables[key] = values if len(values) > 1 else values[0]
        except ValueError:
            # For non-numeric values
            variables[key] = value.split()


D = variables['D']
freq = variables['freq']
typeSrc = variables['typeSrc'][0]
Lx = variables['Lx']
Ly = variables['Ly']
Nx = int(variables['Nx'])
Ny = int(variables['Ny'])
Ntheta = int(variables['Ntheta'])
Nrays = int(variables['Nrays'])
rangeTheta = np.deg2rad(variables['rangeTheta'])
Nphi = int(variables['Nphi'])
rangePhi = np.deg2rad(variables['rangePhi'])
meshMaxSize = variables['meshMaxSize']
typeSurface = variables['typeSurface']
er = variables['er']
tand = variables['tand']
maxRefl = int(variables['Nrefl'])
saveExcels = variables['saveExcels']
plotSurf = variables['plotSurf']
plotDRT = variables['plotDRT']
plotNormals = variables['plotNormals']
plotTubes = variables['plotTubes']

m_max = 10000000                                        #max slope possible
e0 = 8.8541878128e-12                                   #vacuum permitivitty
c0 =299792458                                           #vacuum light speed
wv = c0/freq                                            #wavelength in mm (defined in the paper)
k0 = (2*np.pi/wv)                                       #propagation constant in free space
p = np.linspace(-D, D, 20000)    
nSurfaces = len(typeSurface) + 1   
num_points = 100



match typeSurface[0]:
        case 'cylinder':
            bodies = Cylinder(variables['R1'], variables['h1'], variables['axis1'], variables['center1'], 'lightpink' )

        case 'box':
            bodies = Box(variables['center1'],  variables['axis1'])

        case 'ellipse':
            bodies = Ellipse(variables['center1'], variables['a'], variables['b'], variables['c'])


