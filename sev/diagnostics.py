"""Column diagnostics for 20CRv3. Inputs use K, kg/kg, hPa, m, m/s.

0--6 km wind diagnostics use 10-m wind as the nominal surface (0 m AGL)
and, when present, the model's 12--500 m AGL winds. Bunkers right-mover
uses height-weighted 0--6, 0--0.5, and 5.5--6 km means. SRH is a signed
hodograph line integral and is positive for cyclonic streamwise vorticity.
"""
from __future__ import annotations

import numpy as np

OUTPUT_UNITS = {
    "sbcape": "J kg-1",
    "sbcin": "J kg-1",
    "srh01": "m2 s-2",
    "srh03": "m2 s-2",
    "shear06": "m s-1",
    "storm_u": "m s-1",
    "storm_v": "m s-1",
    "cape_native": "J kg-1",
}


def _layer(z: np.ndarray, u: np.ndarray, v: np.ndarray, bottom: float, top: float):
    """Piecewise-linear profile with exact layer endpoints; no extrapolation."""
    if z.size < 2 or z[0] > bottom + 0.01 or z[-1] < top - 0.01:
        raise ValueError("wind profile does not span requested AGL layer")
    mid = (z > bottom) & (z < top)
    zz = np.r_[bottom, z[mid], top]
    return zz, np.interp(zz, z, u), np.interp(zz, z, v)


def _mean_wind(z: np.ndarray, u: np.ndarray, v: np.ndarray, bottom: float, top: float):
    zz, uu, vv = _layer(z, u, v, bottom, top)
    return np.trapezoid(uu, zz) / (top - bottom), np.trapezoid(vv, zz) / (top - bottom)


def _srh(z: np.ndarray, u: np.ndarray, v: np.ndarray, depth: float, cu: float, cv: float) -> float:
    _, uu, vv = _layer(z, u, v, 0., depth)
    # Equivalent to MetPy's signed polygonal/hodograph SRH integral.
    return float(np.sum((uu[1:] - cu) * (vv[:-1] - cv)
                        - (uu[:-1] - cu) * (vv[1:] - cv)))


def wind_profile(
    p_surface: float, surface_height: float, p_levels: np.ndarray,
    z_msl: np.ndarray, u_pressure: np.ndarray, v_pressure: np.ndarray,
    u10: float, v10: float, low_heights: np.ndarray | None = None,
    u_low: np.ndarray | None = None, v_low: np.ndarray | None = None,
):
    """Build surface-to-upper-level hodograph in meters AGL.

    Pressure-level winds below 500 m are replaced by height-level winds if
    these are supplied; otherwise they are retained. Below-ground levels are
    excluded using BOTH pressure and height. The 10-m wind represents z=0.
    """
    if not np.all(np.isfinite([p_surface, surface_height, u10, v10])):
        raise ValueError("missing surface wind, terrain, or pressure")
    p_levels = np.asarray(p_levels, dtype=float)
    z = np.asarray(z_msl, dtype=float) - surface_height
    up = np.asarray(u_pressure, dtype=float)
    vp = np.asarray(v_pressure, dtype=float)
    valid = (np.isfinite(z) & np.isfinite(up) & np.isfinite(vp)
             & (p_levels < p_surface - 0.1) & (z > 0))
    use_low = (low_heights is not None and u_low is not None and v_low is not None)
    if use_low:
        lh = np.asarray(low_heights, dtype=float)
        lu = np.asarray(u_low, dtype=float)
        lv = np.asarray(v_low, dtype=float)
        keep = np.isfinite(lh) & np.isfinite(lu) & np.isfinite(lv) & (lh > 0)
        valid &= z > 500.  # Keep uniform dense AGL winds below 500 m.
        heights = np.r_[0., lh[keep], z[valid]]
        uu = np.r_[u10, lu[keep], up[valid]]
        vv = np.r_[v10, lv[keep], vp[valid]]
    else:
        heights = np.r_[0., z[valid]]
        uu = np.r_[u10, up[valid]]
        vv = np.r_[v10, vp[valid]]
    ix = np.argsort(heights)
    heights, uu, vv = heights[ix], uu[ix], vv[ix]
    unique = np.r_[True, np.diff(heights) > 0.01]
    return heights[unique], uu[unique], vv[unique]


def kinematic_diagnostics(
    p_surface: float, surface_height: float, p_levels: np.ndarray,
    z_msl: np.ndarray, u_pressure: np.ndarray, v_pressure: np.ndarray,
    u10: float, v10: float, low_heights: np.ndarray | None = None,
    u_low: np.ndarray | None = None, v_low: np.ndarray | None = None,
) -> dict[str, float]:
    """0--1/3 km SRH, 0--6 km bulk shear, Bunkers right-moving motion."""
    z, u, v = wind_profile(p_surface, surface_height, p_levels, z_msl,
                           u_pressure, v_pressure, u10, v10,
                           low_heights, u_low, v_low)
    # Fail closed rather than extrapolating to 6 km AGL.
    mean_u, mean_v = _mean_wind(z, u, v, 0., 6000.)
    low_u, low_v = _mean_wind(z, u, v, 0., 500.)
    high_u, high_v = _mean_wind(z, u, v, 5500., 6000.)
    du, dv = high_u - low_u, high_v - low_v
    norm = float(np.hypot(du, dv))
    if norm < 1.e-8:
        raise ValueError("zero 0--6 km shear vector: undefined Bunkers right mover")
    cu, cv = mean_u + 7.5 * dv / norm, mean_v - 7.5 * du / norm
    return {
        "srh01": _srh(z, u, v, 1000., cu, cv),
        "srh03": _srh(z, u, v, 3000., cu, cv),
        "shear06": float(np.hypot(np.interp(6000., z, u) - np.interp(0., z, u),
                                  np.interp(6000., z, v) - np.interp(0., z, v))),
        "storm_u": float(cu), "storm_v": float(cv),
    }


def surface_cape_cin(
    p_surface: float, t2: float, q2: float,
    p_levels: np.ndarray, t_profile: np.ndarray, q_profile: np.ndarray,
) -> tuple[float, float]:
    """MetPy SBCAPE/CIN from the 2 m parcel, excluding subsurface levels.

    Thermodynamic pressure levels are the shared intersection of temperature
    and specific humidity levels (up to ~100 hPa). Negative/zero humidity is
    clipped to a tiny positive value only for logarithmic dewpoint evaluation.
    """
    if not (np.isfinite(p_surface) and np.isfinite(t2) and np.isfinite(q2)
            and 200. < p_surface < 1100. and 150. < t2 < 350.
            and 0. <= q2 < 0.1):
        raise ValueError("invalid surface thermodynamic state")
    p_levels = np.asarray(p_levels, dtype=float)
    t = np.asarray(t_profile, dtype=float)
    q = np.asarray(q_profile, dtype=float)
    okay = (np.isfinite(p_levels) & np.isfinite(t) & np.isfinite(q)
            & (p_levels < p_surface - 0.1) & (p_levels >= 100.)
            & (t > 150.) & (t < 350.) & (q >= 0.) & (q < 0.1))
    p = np.r_[p_surface, p_levels[okay]]
    T = np.r_[t2, t[okay]]
    Q = np.r_[q2, q[okay]]
    ordering = np.argsort(-p)
    p, T, Q = p[ordering], T[ordering], Q[ordering]
    if p.size < 5 or p[-1] > 200. or not np.all(np.diff(p) < 0.):
        raise ValueError("incomplete sounding to at least 200 hPa")

    # Lazy import: the --skip-cape kinematic benchmark requires only NumPy.
    import metpy.calc as mpcalc
    from metpy.units import units

    pressure = p * units.hectopascal
    temperature = T * units.kelvin
    specific_humidity = np.maximum(Q, 1e-9) * units.dimensionless
    dewpoint = mpcalc.dewpoint_from_specific_humidity(
        pressure=pressure, temperature=temperature,
        specific_humidity=specific_humidity)
    cape, cin = mpcalc.surface_based_cape_cin(pressure, temperature, dewpoint)
    return float(cape.to("joule / kilogram").magnitude), float(cin.to("joule / kilogram").magnitude)
