import json
import numpy as np

# =========================================
# Load JSON input file
# =========================================
with open("In_files/input_iso.json", "r", encoding="utf-8-sig") as f:
    content = f.read()
    f.seek(0)
    cfg = json.load(f)


# =========================================
# Geometry classes
# =========================================
class Cylinder:
    def __init__(self, radius, height, axis, center, er_in, tand_in, color):
        self.type = "cylinder"
        self.radius = radius
        self.height = height
        self.axis = np.array(axis, dtype=float)
        self.center = np.array(center, dtype=float)
        self.er_in = er_in
        self.tand_in = tand_in
        self.color = color


class Box:
    def __init__(self, center, axis, er_in, tand_in, color):
        self.type = "box"
        self.center = np.array(center, dtype=float)
        self.axis = np.array(axis, dtype=float)
        self.er_in = er_in
        self.tand_in = tand_in
        self.color = color


class Ellipse:
    def __init__(self, center, a, b, h, er_in, tand_in, color):
        self.type = "ellipse"
        self.center = np.array(center, dtype=float)
        self.a = a
        self.b = b
        self.h = h
        self.er_in = er_in
        self.tand_in = tand_in
        self.color = color





# =========================================
# Global parameters
# =========================================
D = cfg["D"]
freq = cfg["freq"]
e0 = 8.8541878128e-12
c0 = 299792458
wv = c0 / freq
k0 = 2 * np.pi / wv
p = np.linspace(-D, D, 20000)

typeSrc = cfg["typeSrc"]
theta_pw = cfg["theta"]
phi_pw = cfg["phi"]
Lx = cfg["Lx"]
Ly = cfg["Ly"]
Nx = cfg["Nx"]
Ny = cfg["Ny"]
Nrays = Nx * Ny
Ntheta = int(cfg["Ntheta"])
rangeTheta = np.deg2rad(cfg["rangeTheta"])
Nphi = int(cfg["Nphi"])
rangePhi = np.deg2rad(cfg["rangePhi"])
meshMaxSize = cfg["meshMaxSize"]
Ampl_treshold = cfg["Ampl_treshold"]
Nrefl = int(cfg["Nrefl"])
saveExcels = cfg["saveExcels"]
plotSurf = cfg["plotSurf"]
plotDRT = cfg["plotDRT"]
plotNormals = cfg["plotNormals"]
plotTubes = cfg["plotTubes"]

bckg_er = cfg["bckg_er"]
bckg_tand = cfg["bckg_tand"]

# =========================================
# Geometry parsing
# =========================================

bodies = []
for item in cfg["geometry"]:

    typ = item["type"]
    if typ == "cylinder":
        bodies.append(
            Cylinder(
                radius=item["radius"],
                height=item["height"],
                axis=item["axis"],
                center=item["center"],
                er_in=item["er"],
                tand_in=item["tand"],
                color=item["color"]
            )
        )

    elif typ == "box":
        bodies.append(
            Box(
                center=item["center"],
                axis=item["axis"],
                er_in=item["er"],
                tand_in=item["tand"],
                color=item["color"]
            )
        )

    elif typ == "ellipse":
        bodies.append(
            Ellipse(
                center=item["center"],
                a=item["a"],
                b=item["b"],
                h=item["h"],
                er_in=item["er"],
                tand_in=item["tand"],
                color=item["color"]
            )
        )

    else:
        raise ValueError(f"Unsupported surface type: {typ}")


# Number of surfaces
nStruct = len(bodies)