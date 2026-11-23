"""Direct, read-only access to native NCAR GDEX 20CRv3 3-hourly NetCDF.

Verified GDEX layout:
 /gdex/data/d131003/anl/anl_mean_1900_TMP_pres.nc
 /gdex/data/d131003/invariants/surface_height.nc

A separate dataset is opened per variable, never copied to scratch. Only
selected times, pressure levels, and the CONUS bounding box are read.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xarray as xr

CONUS = (24.0, 50.0, -125.0, -66.0)  # Box, NOT a US political boundary mask.

SOURCES = {
    "ps": "PRES_sfc",
    "t2": "TMP_2m", "q2": "SPFH_2m",
    "t": "TMP_pres", "q": "SPFH_pres",
    "u": "UGRD_pres", "v": "VGRD_pres", "z": "HGT_pres",
    "u10": "UGRD_10m", "v10": "VGRD_10m",
    "uh": "UGRD_height", "vh": "VGRD_height",
    "cape_native": "CAPE_sfc",
}


def _axis(dims: tuple[str, ...], kind: str) -> str:
    aliases = {
        "time": ("time", "valid_time"),
        "lat": ("lat", "latitude"),
        "lon": ("lon", "longitude"),
    }[kind]
    exact = [d for d in dims if d.lower() in aliases]
    if exact:
        return exact[0]
    fuzzy = [d for d in dims if kind in d.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    raise ValueError(f"cannot identify {kind} axis among {dims}")


def _data_var(ds: xr.Dataset) -> xr.DataArray:
    candidates = []
    for name, da in ds.data_vars.items():
        try:
            _axis(da.dims, "lat")
            _axis(da.dims, "lon")
        except ValueError:
            continue
        candidates.append(name)
    if len(candidates) != 1:
        raise ValueError(f"expected one geographic data variable, found {candidates}; data_vars={list(ds.data_vars)}")
    return ds[candidates[0]]


def _vertical(da: xr.DataArray) -> str | None:
    dims = set(da.dims) - {_axis(da.dims, "lat"), _axis(da.dims, "lon")}
    try:
        dims.remove(_axis(da.dims, "time"))
    except ValueError:
        pass
    if len(dims) == 0:
        return None
    if len(dims) > 1:
        # Some NetCDFs have a singleton 2m/10m vertical dimension: these are
        # handled as long as there is only ONE non-horizontal, non-time axis.
        raise ValueError(f"unrecognized additional dimensions in {da.dims}")
    return next(iter(dims))


def _levels(da: xr.DataArray) -> tuple[np.ndarray, str]:
    dim = _vertical(da)
    if dim is None:
        raise ValueError(f"field has no vertical axis: {da.name}")
    coord = da[dim]
    vals = np.asarray(coord.values, dtype=float)
    return vals, str(coord.attrs.get("units", ""))


def _hpa(values: np.ndarray, units: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if units.lower() in {"pa", "pascal", "pascals"} or np.nanmax(values) > 2000.:
        return values / 100.
    return values


def _subset_axis(da: xr.DataArray, indices: np.ndarray, which: str) -> xr.DataArray:
    return da.isel({_axis(da.dims, which): indices})


@dataclass(frozen=True)
class DataLayout:
    latitudes: np.ndarray
    longitudes: np.ndarray
    time: np.ndarray
    p_thermo: np.ndarray
    p_wind: np.ndarray
    low_heights: np.ndarray
    surface_height: np.ndarray  # (nlat, nlon), MSL (m)
    land: np.ndarray | None     # (nlat, nlon), land fraction


class GDEXReader:
    """Open one native annual GDEX file per requested variable.

    A reader instance is confined to the main process: workers receive only
    NumPy column batches, avoiding unsafe sharing of NetCDF/HDF5 handles.
    """
    def __init__(self, root: str | Path, year: int, month: int, *,
                 cape: bool = True, wind: bool = True, height_winds: bool = True,
                 native_cape: bool = False, land_only: bool = False,
                 bbox: tuple[float, float, float, float] = CONUS):
        self.root = Path(root)
        self.year = year
        self.month = month
        self.cape = cape
        self.wind = wind
        self.height_winds = height_winds and wind
        self.native_cape = native_cape
        self.land_only = land_only
        self._stack = ExitStack()
        self.datasets: dict[str, xr.Dataset] = {}
        self.arrays: dict[str, xr.DataArray] = {}
        self.vert_ix: dict[str, np.ndarray] = {}
        self.paths: dict[str, str] = {}
        try:
            names = ["ps"]
            if cape:
                names += ["t2", "q2", "t", "q"]
            if wind:
                names += ["u10", "v10", "u", "v", "z"]
            if self.height_winds:
                names += ["uh", "vh"]
            if native_cape:
                names += ["cape_native"]
            for name in names:
                tag = SOURCES[name]
                path = self.root / "anl" / f"anl_mean_{year}_{tag}.nc"
                if not path.is_file():
                    raise FileNotFoundError(f"GDEX file missing: {path}; check /gdex/data/d131003/anl on Derecho")
                ds = self._stack.enter_context(xr.open_dataset(path, cache=False))
                self.datasets[name] = ds
                self.arrays[name] = _data_var(ds)
                self.paths[name] = str(path)

            ref = self.arrays["ps"]
            latname, lonname, tname = (_axis(ref.dims, k) for k in ("lat", "lon", "time"))
            lat = np.asarray(ref[latname].values, dtype=float)
            lon = np.asarray(ref[lonname].values, dtype=float)
            t = ref[tname]
            timemask = np.asarray(t.dt.month.values == month)
            if not timemask.any():
                raise ValueError(f"No month {month} in surface-pressure file for {year}")
            self.time_ix = np.flatnonzero(timemask)
            self.timesteps = np.asarray(t.values)[self.time_ix]
            south, north, west, east = bbox
            self.lat_ix = np.flatnonzero((lat >= south) & (lat <= north))
            lon180 = (lon + 180) % 360 - 180
            self.lon_ix = np.flatnonzero((lon180 >= west) & (lon180 <= east))
            if not len(self.lat_ix) or not len(self.lon_ix):
                raise ValueError(f"No grid points found for bbox {bbox}")
            self.latitudes = lat[self.lat_ix]
            self.longitudes = lon180[self.lon_ix]
            self.original_longitudes = lon[self.lon_ix]

            for name, da in self.arrays.items():
                la, lo, ti = (_axis(da.dims, k) for k in ("lat", "lon", "time"))
                if not (np.allclose(da[la].values, lat) and np.allclose(da[lo].values, lon)):
                    raise ValueError(f"Grid mismatch in {name}: {self.paths[name]}")
                if not np.array_equal(np.asarray(da[ti].values).astype(str),
                                      np.asarray(t.values).astype(str)):
                    raise ValueError(f"Time axis mismatch in {name}: {self.paths[name]}")

            self.p_thermo = np.empty(0)
            self.p_wind = np.empty(0)
            self.low_heights = np.empty(0)
            if cape:
                self.p_thermo = self._match_levels(("t", "q"), min_level=100.)
            if wind:
                self.p_wind = self._match_levels(("u", "v", "z"), min_level=300.)
            if self.height_winds:
                self.low_heights = self._match_levels(("uh", "vh"), min_level=0., height=True)
            self.surface_height = self._static("surface_height.nc")
            self.land = self._static("land_mask.nc") if land_only else None
        except BaseException:
            self._stack.close()
            raise

    def _match_levels(self, names: tuple[str, ...], *, min_level: float,
                      height: bool = False) -> np.ndarray:
        common = None
        for name in names:
            values, unit = _levels(self.arrays[name])
            scaled = values if height else _hpa(values, unit)
            common = scaled if common is None else np.intersect1d(common, scaled)
        assert common is not None
        common = common[(common >= min_level) & (common <= (500. if height else 1100.))]
        common = np.sort(common) if height else np.sort(common)[::-1]
        if common.size == 0:
            raise ValueError(f"no shared vertical levels among {names}")
        for name in names:
            vals, units = _levels(self.arrays[name])
            vals = vals if height else _hpa(vals, units)
            # Avoid fragile floating exact equality on unit-converted coordinates.
            ix = [int(np.argmin(np.abs(vals - target))) for target in common]
            if not np.allclose(vals[ix], common, rtol=1e-7, atol=1e-4):
                raise ValueError(f"level mismatch in {name}")
            self.vert_ix[name] = np.asarray(ix, dtype=int)
        return common

    def _static(self, filename: str) -> np.ndarray:
        path = self.root / "invariants" / filename
        if not path.is_file():
            raise FileNotFoundError(f"required invariant GDEX file missing: {path}")
        ds = self._stack.enter_context(xr.open_dataset(path, cache=False))
        da = _data_var(ds)
        da = _subset_axis(_subset_axis(da, self.lat_ix, "lat"), self.lon_ix, "lon")
        la, lo = (_axis(da.dims, k) for k in ("lat", "lon"))
        da = da.transpose(la, lo)
        result = np.asarray(da.values, dtype=np.float32)
        if result.shape != (len(self.lat_ix), len(self.lon_ix)):
            raise ValueError(f"shape mismatch for invariant {filename}")
        return result

    def describe(self) -> dict:
        return {
            "root": str(self.root), "year": self.year, "month": self.month,
            "nlat": len(self.lat_ix), "nlon": len(self.lon_ix),
            "npoints": len(self.lat_ix) * len(self.lon_ix),
            "ntimes": len(self.time_ix),
            "first_time": str(self.timesteps[0]),
            "last_time": str(self.timesteps[-1]),
            "p_thermo_hPa": self.p_thermo.tolist(),
            "p_wind_hPa": self.p_wind.tolist(),
            "low_heights_m": self.low_heights.tolist(),
            "paths": self.paths,
            "variables": {name: {"variable": da.name, "dims": da.dims,
                                 "shape": da.shape, "units": da.attrs.get("units", "")}
                          for name, da in self.arrays.items()},
        }

    def load_block(self, begin: int, end: int) -> dict[str, np.ndarray]:
        """Return time-major blocks with pressure axis second and flattened XY.

        The file reads are serialized in the parent process. Multiprocessing
        workers never touch shared NetCDF handles.
        """
        if not 0 <= begin < end <= len(self.time_ix):
            raise ValueError(f"invalid selected time range [{begin}, {end})")
        time_indices = self.time_ix[begin:end]
        data = {}
        for name, da in self.arrays.items():
            lat, lon, time = (_axis(da.dims, k) for k in ("lat", "lon", "time"))
            selection = {time: time_indices, lat: self.lat_ix, lon: self.lon_ix}
            vert = _vertical(da)
            if name in self.vert_ix:
                selection[vert] = self.vert_ix[name]
            elif vert is not None:
                if da.sizes[vert] != 1:
                    raise ValueError(f"unexpected vertical dimension for {name}: {da.sizes}")
                selection[vert] = [0]
            subset = da.isel(selection)
            if vert is not None and name not in self.vert_ix:
                subset = subset.isel({vert: 0}, drop=True)
                vert = None
            order = [time] + ([vert] if vert is not None else []) + [lat, lon]
            raw = np.asarray(subset.transpose(*order).values, dtype=np.float32)
            if vert is None:
                data[name] = raw.reshape(raw.shape[0], -1)
            else:
                data[name] = raw.reshape(raw.shape[0], raw.shape[1], -1)
        if np.nanmedian(data["ps"]) > 2000.:
            data["ps"] = data["ps"] / 100.  # Pa -> hPa
        data["surface_height"] = self.surface_height.ravel()
        if self.land is not None:
            data["land"] = self.land.ravel()
        return data

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
