import numpy as np
from scipy.interpolate import LinearNDInterpolator


#===========================================================================================================
# Read CST far-field file
def read_cst_farfield(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines()[2:]:              # Skip header
            parts = line.split()
            if len(parts) == 8:
                data.append([float(x) for x in parts])

    data = np.array(data)

    return {
        "theta": data[:, 0],                        # [deg]
        "phi":   data[:, 1],                        # [deg]
        "Eth":   data[:, 3] * np.exp(1j*np.deg2rad(data[:, 4])),  # Complex Eθ
        "Eph":   data[:, 5] * np.exp(1j*np.deg2rad(data[:, 6]))   # Complex Eφ
    }
#===========================================================================================================


#===========================================================================================================
# Build interpolator (normalized)
def build_farfield_interpolator(filename):
    raw = read_cst_farfield(filename)

    # Normalize
    E_mag = np.sqrt(np.abs(raw["Eth"])**2 + np.abs(raw["Eph"])**2)
    Eth = raw["Eth"] / np.max(E_mag)
    Eph = raw["Eph"] / np.max(E_mag)

    pts = np.column_stack([raw["theta"], raw["phi"]])  # (θ,φ) grid

    return {
        "Eth_r": LinearNDInterpolator(pts, Eth.real),  # Real part
        "Eth_i": LinearNDInterpolator(pts, Eth.imag),  # Imag part
        "Eph_r": LinearNDInterpolator(pts, Eph.real),
        "Eph_i": LinearNDInterpolator(pts, Eph.imag),
    }
#===========================================================================================================


#===========================================================================================================
# Evaluate complex field at (θ, φ) [deg]
def evaluate_farfield(ff, theta_deg, phi_deg):
    theta_deg = np.atleast_1d(theta_deg).astype(float)
    phi_deg   = np.mod(np.atleast_1d(phi_deg).astype(float), 360.0)

    Eth = ff["Eth_r"](theta_deg, phi_deg) + 1j * ff["Eth_i"](theta_deg, phi_deg)
    Eph = ff["Eph_r"](theta_deg, phi_deg) + 1j * ff["Eph_i"](theta_deg, phi_deg)

    return Eth, Eph
#===========================================================================================================


#===========================================================================================================
# Convert to Cartesian components
def farfield_to_cartesian(ff, theta, phi):
    theta = np.atleast_1d(theta)                    # [rad]
    phi   = np.atleast_1d(phi)

    Eth, Eph = evaluate_farfield(
        ff,
        np.rad2deg(theta),
        np.rad2deg(phi)
    )

    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi),   np.sin(phi)

    Ex = Eth * ct * cp - Eph * sp                  # x-component
    Ey = Eth * ct * sp + Eph * cp                  # y-component
    Ez = -Eth * st                                 # z-component

    return Ex, Ey, Ez
#===========================================================================================================

# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------
# ff = build_farfield_interpolator("Sources/wvgd_src_lin.txt")
# Ex, Ey, Ez = farfield_to_cartesian(ff, theta, phi)

