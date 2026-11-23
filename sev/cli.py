"""Monthly CONUS diagnostics and profiling from direct-read native GDEX files.

Run ``python -m sev.cli --help``. No remote downloads or Dask scheduler are
required; NetCDF reads happen in the parent process and column calculations
are spread over local process workers on the allocated Derecho node.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.util
import multiprocessing
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

from .diagnostics import OUTPUT_UNITS, kinematic_diagnostics, surface_cape_cin
from .gdex import GDEXReader


def _tile_work(task):
    """One time and one contiguous collection of columns, no I/O."""
    time_index, point_indices, tile, pthermo, pwind, low_heights, cape, wind, native = task
    out_names = ((["sbcape", "sbcin"] if cape else [])
                 + (["srh01", "srh03", "shear06", "storm_u", "storm_v"] if wind else [])
                 + (["cape_native"] if native else []))
    out = {name: np.full(len(point_indices), np.nan, dtype=np.float32) for name in out_names}
    counts = {"cape_valid": 0, "cape_invalid": 0,
              "wind_valid": 0, "wind_invalid": 0, "cape_seconds": 0., "wind_seconds": 0.}
    examples: list[str] = []
    for j, idx in enumerate(point_indices):
        if "land" in tile and not np.isfinite(tile["land"][j]):
            continue
        if "land" in tile and tile["land"][j] < 0.5:
            continue
        ps = float(tile["ps"][j])
        terrain = float(tile["surface_height"][j])
        if native:
            out["cape_native"][j] = tile["cape_native"][j]
        if cape:
            t0 = perf_counter()
            try:
                value, inhibition = surface_cape_cin(
                    ps, float(tile["t2"][j]), float(tile["q2"][j]),
                    pthermo, tile["t"][:, j], tile["q"][:, j])
                out["sbcape"][j] = value
                out["sbcin"][j] = inhibition
                counts["cape_valid"] += 1
            except (ValueError, ArithmeticError, FloatingPointError, IndexError) as exc:
                counts["cape_invalid"] += 1
                if len(examples) < 3:
                    examples.append(f"cape column {idx}: {exc}")
            counts["cape_seconds"] += perf_counter() - t0
        if wind:
            t0 = perf_counter()
            try:
                kwargs = {}
                if "uh" in tile:
                    kwargs = {"low_heights": low_heights,
                              "u_low": tile["uh"][:, j], "v_low": tile["vh"][:, j]}
                values = kinematic_diagnostics(
                    ps, terrain, pwind, tile["z"][:, j],
                    tile["u"][:, j], tile["v"][:, j],
                    float(tile["u10"][j]), float(tile["v10"][j]), **kwargs)
                for name, value in values.items():
                    out[name][j] = value
                counts["wind_valid"] += 1
            except (ValueError, ArithmeticError, FloatingPointError, IndexError) as exc:
                counts["wind_invalid"] += 1
                if len(examples) < 3:
                    examples.append(f"wind column {idx}: {exc}")
            counts["wind_seconds"] += perf_counter() - t0
    return time_index, point_indices, out, counts, examples


def _tasks(block, selected, pthermo, pwind, low_heights, *, cape, wind, native, tile_size):
    nt = block["ps"].shape[0]
    for t in range(nt):
        for start in range(0, len(selected), tile_size):
            point_indices = selected[start:start + tile_size]
            # Only serialize a tiny 1-time, tile-sized view to each worker.
            tile = {}
            for key, array in block.items():
                if array.ndim == 3:
                    tile[key] = np.ascontiguousarray(array[t][:, point_indices])
                elif array.ndim == 2:
                    tile[key] = np.ascontiguousarray(array[t, point_indices])
                else:
                    tile[key] = np.ascontiguousarray(array[point_indices])
            yield (t, point_indices, tile, pthermo, pwind, low_heights, cape, wind, native)


def _write(ds: xr.Dataset, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.nc")
    if importlib.util.find_spec("netCDF4") is not None:
        encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 3,
                           "_FillValue": np.float32(np.nan)} for name in ds.data_vars}
        ds.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    else:
        # NetCDF3 fallback only for tiny local synthetic tests.
        ds.to_netcdf(temporary, engine="scipy")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/gdex/data/d131003", help="native NCAR GDEX mount")
    parser.add_argument("--year", type=int, default=1900)
    parser.add_argument("--month", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--time-block", type=int, default=8, help="3-hourly times loaded per I/O block")
    parser.add_argument("--max-times", type=int, default=0,
                        help="first N 3-hourly times (0 = complete month)")
    parser.add_argument("--max-points", type=int, default=0,
                        help="sample N locations across bounding box (0 = complete region)")
    parser.add_argument("--land-only", action="store_true",
                        help="skip ocean points, NOT an exact US political boundary")
    parser.add_argument("--no-height-winds", action="store_true",
                        help="omit model 12--500 m winds (pressure-level-only hodograph)")
    parser.add_argument("--skip-cape", action="store_true")
    parser.add_argument("--skip-wind", action="store_true")
    parser.add_argument("--native-cape", action="store_true",
                        help="also read archived ensemble-mean CAPE_sfc for comparison")
    parser.add_argument("--inspect", action="store_true", help="report file paths/coordinates, no computation")
    parser.add_argument("--no-write", action="store_true", help="benchmark without output NetCDF")
    parser.add_argument("--output-dir", default="results", help="diagnostic NetCDF and benchmark JSON directory")
    args = parser.parse_args(argv)
    if args.month not in range(1, 13) or args.workers < 1 or args.time_block < 1 or args.tile_size < 1:
        parser.error("month=1..12; workers, time-block, tile-size must be >=1")
    if args.max_times < 0 or args.max_points < 0:
        parser.error("max-times and max-points must be nonnegative")
    if args.skip_cape and args.skip_wind and not args.native_cape:
        parser.error("no diagnostics selected")
    if not args.inspect and not args.skip_cape and importlib.util.find_spec("metpy") is None:
        parser.error("MetPy not installed; install the SEV environment or run --skip-cape")

    output_dir = Path(args.output_dir)
    stem = f"sev_{args.year}{args.month:02d}"
    # Benchmark/small sample output cannot silently clobber a complete month.
    if args.max_times or args.max_points:
        stem += f"_benchmark_t{args.max_times or 'all'}_p{args.max_points or 'all'}"
    if args.skip_cape:
        stem += "_no_cape"
    if args.skip_wind:
        stem += "_no_wind"
    output = output_dir / f"{stem}.nc"
    report_path = output_dir / f"{stem}.json"
    begun = perf_counter()
    with GDEXReader(args.root, args.year, args.month, cape=not args.skip_cape,
                    wind=not args.skip_wind, height_winds=not args.no_height_winds,
                    native_cape=args.native_cape, land_only=args.land_only) as reader:
        desc = reader.describe()
        if args.inspect:
            print(json.dumps(desc, indent=2, default=str), flush=True)
            return 0
        n_times = min(args.max_times or len(reader.time_ix), len(reader.time_ix))
        npts = desc["npoints"]
        selected = np.arange(npts)
        if args.land_only:
            selected = selected[np.isfinite(reader.land.ravel()[selected]) &
                                (reader.land.ravel()[selected] >= 0.5)]
        if args.max_points and len(selected) > args.max_points:
            selected = selected[np.linspace(0, len(selected) - 1, args.max_points, dtype=int)]
        if len(selected) == 0:
            raise RuntimeError("No points selected")
        wind = not args.skip_wind
        cape = not args.skip_cape
        keys = ((["sbcape", "sbcin"] if cape else [])
                + (["srh01", "srh03", "shear06", "storm_u", "storm_v"] if wind else [])
                + (["cape_native"] if args.native_cape else []))
        fields = {key: np.full((n_times, npts), np.nan, dtype=np.float32) for key in keys}
        stats = {"cape_valid": 0, "cape_invalid": 0, "wind_valid": 0,
                 "wind_invalid": 0, "cape_seconds": 0., "wind_seconds": 0.}
        io_s = 0.
        worker_wall_s = 0.
        error_examples: list[str] = []
        print(f"Dataset: {desc['nlat']} x {desc['nlon']} points, "
              f"{n_times} times, {len(selected)} selected columns per timestep; "
              f"{args.workers} workers", flush=True)
        print(f"Thermodynamic levels: {len(reader.p_thermo)}; wind levels: "
              f"{len(reader.p_wind)}; near-surface wind levels: "
              f"{len(reader.low_heights)}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for begin in range(0, n_times, args.time_block):
                end = min(begin + args.time_block, n_times)
                io0 = perf_counter()
                block = reader.load_block(begin, end)
                elapsed_io = perf_counter() - io0
                io_s += elapsed_io
                comp0 = perf_counter()
                tasks = _tasks(block, selected, reader.p_thermo, reader.p_wind,
                               reader.low_heights, cape=cape, wind=wind,
                               native=args.native_cape, tile_size=args.tile_size)
                # Map returns tasks in order, but each task's payload carries its
                # explicit time/point indices, avoiding ordering assumptions.
                for time_ix, indices, results, counts, examples in pool.map(_tile_work, tasks):
                    for name, value in results.items():
                        fields[name][begin + time_ix, indices] = value
                    for key, value in counts.items():
                        stats[key] += value
                    if len(error_examples) < 10:
                        error_examples.extend(examples[:10 - len(error_examples)])
                elapsed_comp = perf_counter() - comp0
                worker_wall_s += elapsed_comp
                print(f"block [{begin}:{end}] read={elapsed_io:.2f}s compute={elapsed_comp:.2f}s "
                      f"cape_valid={stats['cape_valid']} wind_valid={stats['wind_valid']}",
                      flush=True)
        coords = {"time": reader.timesteps[:n_times],
                  "latitude": reader.latitudes, "longitude": reader.longitudes}
        output_vars = {}
        for name, arr in fields.items():
            array = arr.reshape(n_times, desc["nlat"], desc["nlon"])
            output_vars[name] = (("time", "latitude", "longitude"), array,
                                 {"units": OUTPUT_UNITS[name]})
        ds = xr.Dataset(output_vars, coords=coords, attrs={
            "description": "20CRv3 historical severe-weather environments; analysis ensemble-mean profiles",
            "data_source": "NCAR GDEX d131003: annual 3-hourly Gaussian T254 analysis means",
            "bounding_box": "24-50N, 125-66W; includes parts of Canada/Mexico; not a political mask",
            "parcel": "2-m temperature and specific humidity at model surface pressure",
            "storm_motion": "Bunkers (2000) right mover, 7.5 m/s deviation; height-weighted winds",
            "surface_wind": "10-m wind treated as z=0 m AGL for kinematic layers",
            "height_definition": "pressure-level geopotential height minus invariant surface_height",
            "low_level_winds": "12-500 m model AGL" if reader.height_winds else "pressure-level only",
            "metpy_sbcape": "computed separately from ensemble mean T, q, ps",
            "native_cape": "archived ensemble mean of model CAPE diagnostic" if args.native_cape else "not loaded",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_s = 0.
        if not args.no_write:
            t0 = perf_counter()
            _write(ds, output)
            write_s = perf_counter() - t0
        total_s = perf_counter() - begun
        selected_soundings = n_times * len(selected)
        throughput = selected_soundings / worker_wall_s if worker_wall_s else None
        report = {
            "arguments": vars(args), "n_gridpoints": npts,
            "n_selected_gridpoints": len(selected), "n_times": n_times,
            "n_columns_attempted": selected_soundings,
            "io_wall_seconds": io_s, "worker_parallel_wall_seconds": worker_wall_s,
            "compute_aggregate_worker_seconds": stats["cape_seconds"] + stats["wind_seconds"],
            "output_write_seconds": write_s, "total_wall_seconds": total_s,
            "throughput_columns_per_wall_second": throughput,
            "cape_seconds_per_column": (stats["cape_seconds"] /
                                        (stats["cape_valid"] + stats["cape_invalid"]))
                if (stats["cape_valid"] + stats["cape_invalid"]) else None,
            "wind_seconds_per_column": (stats["wind_seconds"] /
                                        (stats["wind_valid"] + stats["wind_invalid"]))
                if (stats["wind_valid"] + stats["wind_invalid"]) else None,
            "counts": {k: v for k, v in stats.items() if not k.endswith("seconds")},
            "errors_example": error_examples,
            "files": desc["paths"], "output_file": str(output) if not args.no_write else None,
            "output_bytes": output.stat().st_size if not args.no_write else None,
            "benchmark_note": "CPU seconds are process-wall time summed across serial column kernels;"
                              "  use actual Derecho timings to extrapolate.",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({key: report[key] for key in (
            "total_wall_seconds", "io_wall_seconds", "worker_parallel_wall_seconds",
            "throughput_columns_per_wall_second", "cape_seconds_per_column",
            "wind_seconds_per_column", "counts", "output_file")}, indent=2), flush=True)
        print(f"Benchmark report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
