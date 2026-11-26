"""MILESTONE 4 — current in tissue.  YOUR WORK, with the linear algebra provided.

We solve  div(sigma grad V) = 0  on an (r, z) grid, which is legal because
everything is symmetric about the catheter axis: nothing depends on the angle
around it, so a 2-D grid describes a 3-D problem.

Geometry: the electrode is a sphere of radius a centred at z = a - depth, the
tissue interface is the plane z = 0, myocardium fills z < 0 and blood z > 0.
Cells at or beyond ``domain_radius`` from the origin are held at V = 0.

TWO FUNCTIONS ARE GIVEN TO YOU: build_grid, which is bookkeeping, and
assemble_and_solve, which builds the matrix and solves it.  Writing a sparse
solver is software work, not electromagnetics.  Everything else is yours, and
all of it is physics.

    python check.py 4
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import constants as C
from .geometry import AblationSettings


# ===========================================================================
#  PROVIDED — read these two, do not edit them
# ===========================================================================

def build_grid(settings: AblationSettings = None):
    """PROVIDED.  Cell-centred axisymmetric grid.

    Returns a dict with:
        h        cell size, m
        nr, nz   number of cells across and up
        r, z     1-D arrays of cell-centre coordinates
        R, Z     (nr, nz) tables of those coordinates, one entry per cell
        volume   (nr, nz) cell volumes.  A cell at radius r is a ring, so its
                 volume is 2 pi r h^2, not h^3.
    """
    s = settings or AblationSettings()
    h = s.cell_size
    nr = int(round(s.domain_radius / h))
    nz = int(round(2.0 * s.domain_half_height / h))
    r = (np.arange(nr) + 0.5) * h
    z = -s.domain_half_height + (np.arange(nz) + 0.5) * h
    R, Z = np.meshgrid(r, z, indexing="ij")
    return {"h": h, "nr": nr, "nz": nz, "r": r, "z": z, "R": R, "Z": Z,
            "volume": 2.0 * np.pi * R * h * h}


def assemble_and_solve(grid, g_r, g_z, fixed, fixed_values):
    """PROVIDED.  Solve the network of conductances for the cell voltages.

    Think of every cell as a node and every face as a resistor.  Kirchhoff's
    current law at each free node is one equation; this builds that system and
    solves it.

    g_r          (nr-1, nz) conductance of each radial face
    g_z          (nr, nz-1) conductance of each axial face
    fixed        (nr, nz) boolean, True where the voltage is known
    fixed_values (nr, nz) the known voltages (anything where fixed is False is
                 ignored)
    returns      (nr, nz) the solved potential
    """
    nr, nz = grid["nr"], grid["nz"]
    idx = np.arange(nr * nz).reshape(nr, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros(nr * nz)

    def add_face(ia, ja, ib, jb, g):
        a, b = idx[ia, ja].ravel(), idx[ib, jb].ravel()
        gg = g.ravel()
        rows.extend([a, b]); cols.extend([b, a]); vals.extend([-gg, -gg])
        np.add.at(diag, a, gg); np.add.at(diag, b, gg)

    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nz), indexing="ij")
    add_face(ii, jj, ii + 1, jj, g_r)
    ii, jj = np.meshgrid(np.arange(nr), np.arange(nz - 1), indexing="ij")
    add_face(ii, jj, ii, jj + 1, g_z)

    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nr * nz, nr * nz)).tocsr() + sp.diags(diag)
    f = fixed.ravel()
    b = -A[:, f] @ fixed_values.ravel()[f]
    v = np.zeros(nr * nz)
    v[f] = fixed_values.ravel()[f]
    v[~f] = spla.spsolve(A[~f][:, ~f].tocsc(), b[~f])
    return v.reshape(nr, nz)


# ===========================================================================
#  YOUR WORK
# ===========================================================================

def conductivity_map(grid, settings: AblationSettings = None):
    """(nr, nz) table of sigma: myocardium below the interface, blood above."""
    s = settings or AblationSettings()
    return np.where(grid["Z"] < 0.0, s.sigma_myocardium, s.sigma_blood)


def electrode_mask(grid, settings: AblationSettings = None):
    """(nr, nz) boolean: True in cells inside the spherical electrode."""
    s = settings or AblationSettings()
    d = np.hypot(grid["R"], grid["Z"] - s.electrode_centre_z)
    return d <= s.electrode_radius


def far_boundary_mask(grid, settings: AblationSettings = None):
    """(nr, nz) boolean: True in cells at or beyond the grounded shell."""
    s = settings or AblationSettings()
    return np.hypot(grid["R"], grid["Z"]) >= s.domain_radius


def harmonic_mean(sigma_a, sigma_b):
    """JOB  the effective conductivity of a face between two cells of different sigma."""
    return 2.0 * sigma_a * sigma_b / (sigma_a + sigma_b)


def face_conductances(sigma, grid):
    """Conductance of every face.  Returns (g_r, g_z)."""
    h = grid["h"]
    R = grid["R"]
    r_face = 0.5 * (R[:-1, :] + R[1:, :])
    g_r = harmonic_mean(sigma[:-1, :], sigma[1:, :]) * (2.0 * np.pi * r_face * h) / h
    g_z = harmonic_mean(sigma[:, :-1], sigma[:, 1:]) * (2.0 * np.pi * R[:, :-1] * h) / h
    return g_r, g_z


def solve_potential(grid, sigma, settings: AblationSettings = None):
    """(nr, nz) potential, volts.  This one just wires your pieces together."""
    s = settings or AblationSettings()
    elec = electrode_mask(grid, s)
    far = far_boundary_mask(grid, s)
    fixed = elec | far
    fixed_values = np.where(elec, s.voltage, 0.0)
    g_r, g_z = face_conductances(sigma, grid)
    return assemble_and_solve(grid, g_r, g_z, fixed, fixed_values)


def face_powers(v, sigma, grid):
    """Power dissipated in each face, G * (dV)^2.  Returns (p_r, p_z)."""
    g_r, g_z = face_conductances(sigma, grid)
    p_r = g_r * (v[:-1, :] - v[1:, :])**2
    p_z = g_z * (v[:, :-1] - v[:, 1:])**2
    return p_r, p_z


def power_density(v, sigma, grid, settings: AblationSettings = None):
    """(nr, nz) table of p = sigma |E|^2 in W/m^3."""
    s = settings or AblationSettings()
    p_r, p_z = face_powers(v, sigma, grid)
    cell = np.zeros_like(v)
    cell[:-1, :] += 0.5 * p_r
    cell[1:, :] += 0.5 * p_r
    cell[:, :-1] += 0.5 * p_z
    cell[:, 1:] += 0.5 * p_z
    out = cell / grid["volume"]
    out[electrode_mask(grid, s)] = 0.0
    return out


def total_power(v, sigma, grid, region=None):
    """Total dissipated power, W."""
    p_r, p_z = face_powers(v, sigma, grid)
    if region is not None:
        p_dens = power_density(v, sigma, grid)
        return float(np.sum(p_dens[region] * grid["volume"][region]))
    return float(np.sum(p_r) + np.sum(p_z))


def electrode_current(v, sigma, grid, settings: AblationSettings = None):
    """Current delivered by the electrode, A."""
    s = settings or AblationSettings()
    if s.voltage == 0:
        return 0.0
    return total_power(v, sigma, grid) / s.voltage


def impedance(v, sigma, grid, settings: AblationSettings = None):
    """Z = V0 / I, ohms.  One line.  This is the number electrophysiologists watch."""
    s = settings or AblationSettings()
    I = electrode_current(v, sigma, grid, s)
    return float(s.voltage / I) if I != 0 else float('inf')


def power_fraction_myocardium(v, sigma, grid):
    """Share of the delivered power landing in tissue, between 0 and 1."""
    p_dens = power_density(v, sigma, grid)
    vol = grid["volume"]
    myo = grid["Z"] < 0.0
    total = np.sum(p_dens * vol)
    return float(np.sum(p_dens[myo] * vol[myo]) / total) if total != 0 else 0.0


def lesion_volume(v, sigma, grid, settings: AblationSettings = None):
    """Volume of myocardium above the power-density threshold, m^3."""
    s = settings or AblationSettings()
    p = power_density(v, sigma, grid, s)
    hot = (p > s.lesion_threshold) & (grid["Z"] < 0.0) & (~electrode_mask(grid, s))
    return float(np.sum(grid["volume"][hot]))


def analytic_resistance(settings: AblationSettings = None, straddling=True):
    """The closed form your grid answer is checked against, ohms."""
    s = settings or AblationSettings()
    num = (1.0 / s.electrode_radius) - (1.0 / s.domain_radius)
    if straddling:
        return float(num / (2.0 * np.pi * (s.sigma_myocardium + s.sigma_blood)))
    else:
        return float(num / (4.0 * np.pi * s.sigma_blood))


def solve(settings: AblationSettings = None):
    """Run the whole chain once and return a dict."""
    s = settings or AblationSettings()
    grid = build_grid(s)
    sigma = conductivity_map(grid, s)
    v = solve_potential(grid, sigma, s)
   
    return {
        "grid": grid,
        "sigma": sigma,
        "v": v,
        "impedance": impedance(v, sigma, grid, s),
        "current": electrode_current(v, sigma, grid, s),
        "power": total_power(v, sigma, grid),
        "fraction_myocardium": power_fraction_myocardium(v, sigma, grid),
        "lesion_volume": lesion_volume(v, sigma, grid, s)
    }
